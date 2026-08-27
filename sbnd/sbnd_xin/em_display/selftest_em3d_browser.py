"""doc pr/114 round 3 -- drive the 3-D view in a REAL browser.

The rest of the round's verification is static: selftest_em_display.py lints the
CustomJS (every free name supplied through `args`, brackets balanced, the guard
reading the live gesture state) and tests the Python mirrors of the geometry, but
it cannot execute a line of JS.  This script can.

Round 3 first concluded the browser code was untestable here, because there is no
node, deno, esprima, js2py, dukpy or quickjs anywhere in the tree -- true, and the
wrong conclusion.  A *JS engine* is absent; **playwright's bundled chromium is
installed** (~/.cache/ms-playwright/chromium-1228), and a headless browser runs
the code exactly the way the owner's will.  "No interpreter for this language" is
not the same question as "nothing here can run this code".

Two things it proves that nothing else can:

  * the JS and Python projections AGREE.  There are two mirrors of one formula
    (em3d.project and em3d.JS_PROJECT); this reads the browser's own u/v back out
    of the ColumnDataSource after a camera change and compares them, point by
    point, against Python's.  That is the drift risk of the whole design, closed.
  * the gestures actually do what they claim.  A synthetic drag really produces
    Pan events with no drag tool active, the rotate handler really fires,
    shift+drag really pans, the wheel really zooms, and Box Select really
    suspends rotation -- each of which rests on a bokehjs internal that was read
    rather than executed until now.

Bokeh compiles a CustomJS body LAZILY, on first execution, so merely loading the
page does not prove the code parses.  Every check below triggers a handler.

Run:   python em_display/selftest_em3d_browser.py [--port 5029] [--headed]
       (starts and stops its own bokeh server; needs no scan tag written)
Skips cleanly with rc=0 and a loud message if playwright's chromium is absent.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5029)
ap.add_argument("--headed", action="store_true")
opts = ap.parse_args()

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed -- SKIPPING the browser test (not a failure).")
    sys.exit(0)

import em3d as D3  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print("%-58s %s %s" % (name, "PASS" if cond else "**FAIL**", detail))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
URL = "http://localhost:%d/em_display_viewer" % opts.port
srv = subprocess.Popen(
    [os.path.join(SX, "em_display", "serve_em_display.sh"), str(opts.port),
     "--scan-tag", "em3dbrowsertest"],
    cwd=SX, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    preexec_fn=os.setsid)


def stop_server():
    try:
        os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


try:
    import urllib.request
    for _ in range(60):
        time.sleep(1.0)
        try:
            urllib.request.urlopen(URL, timeout=3).read()
            break
        except Exception:
            continue
    else:
        print("**FAIL** server never came up on %d" % opts.port)
        stop_server()
        sys.exit(1)

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=not opts.headed)
        except Exception as exc:
            print("chromium unavailable (%s) -- SKIPPING (not a failure)."
                  % str(exc).splitlines()[0][:90])
            stop_server()
            sys.exit(0)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors = []
        page.on("console", lambda m: errors.append("%s: %s" % (m.type, m.text))
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function(
            "() => window.Bokeh && Bokeh.documents && Bokeh.documents.length > 0",
            timeout=60000)
        # The whole layout has to be laid out before any canvas has a size.
        page.wait_for_timeout(4000)

        def js(expr):
            return page.evaluate("() => { const doc = Bokeh.documents[0];"
                                 " const M = (n) => doc.get_model_by_name(n);"
                                 " return (%s); }" % expr)

        check("page loads and a Bokeh document attaches", js("doc != null"))
        check("  ... with no console errors on load", not errors,
              "; ".join(errors[:2]))

        n_cloud = js("M('cloud_src').data.x.length")
        check("the charge cloud reached the browser", n_cloud > 1000,
              "%s points" % n_cloud)

        # ------------------------------------------------------------------
        # the 3-D canvas
        # ------------------------------------------------------------------
        # Bokeh 3 renders every view inside an OPEN shadow root, five deep here, so
        # document.querySelectorAll('canvas') finds nothing at all.  Walk it.
        RECTS = """(() => {
            const out = [], walk = (root) => {
                for (const el of root.querySelectorAll('*')) {
                    if (el.tagName === 'CANVAS') {
                        const r = el.getBoundingClientRect();
                        out.push({x: r.x, y: r.y, w: r.width, h: r.height});
                    }
                    if (el.shadowRoot) walk(el.shadowRoot);
                }
            };
            walk(document);
            return out;
        })()"""

        # Every rect filter below is keyed to a width READ OFF THE MODEL, never a
        # literal.  Round 4 both resized the 3-D panel and added a size selector,
        # and the previous version of this file filtered on `w > 410 && w < 425`
        # -- constants from the round-3 layout.  A filter that matches nothing
        # does not fail, it passes vacuously, which is the worst way for a test
        # to break.
        def by_size(model, tol=8):
            return page.evaluate(
                "() => { const doc = Bokeh.documents[0];"
                " const m = doc.get_model_by_name('%s');"
                " return %s.filter(r => Math.abs(r.w - m.width) < %d"
                " && Math.abs(r.h - m.height) < %d)"
                " .filter((r, i, a) => a.findIndex("
                "     q => Math.abs(q.x - r.x) < 2 && Math.abs(q.y - r.y) < 2)"
                "   === i); }" % (model, RECTS, tol, tol))

        w3d = js("M('f3d').width")
        boxes = by_size("f3d")
        box = boxes[0] if boxes else None
        check("the 3-D canvas is laid out at the size its model asks for",
              box is not None and abs(box["w"] - w3d) < 8, "model %s, dom %s"
              % (w3d, box))
        cx = box["x"] + box["w"] / 2.0
        cy = box["y"] + box["h"] / 2.0

        # ------------------------------------------------------------------
        # round 4 request 1: the app uses the WIDTH of the screen
        # ------------------------------------------------------------------
        # The acceptance plot lives in the right-hand column, so if the layout
        # really is two columns its canvas starts to the right of where the 3-D
        # canvas ends.  Stacked in one column (round 3) it would start at the
        # same left edge.
        accb = by_size("acc")
        check("the right-hand column really is to the RIGHT of the 3-D view",
              bool(accb) and accb[0]["x"] > box["x"] + box["w"],
              ("3-D spans x %.0f..%.0f, acceptance plot starts at x %.0f"
               % (box["x"], box["x"] + box["w"], accb[0]["x"])) if accb
               else "acceptance canvas not found")

        # ------------------------------------------------------------------
        # a preset button: compiles and runs JS_REDRAW, and reports the camera
        # ------------------------------------------------------------------
        page.get_by_role("button", name="x-z", exact=True).click()
        page.wait_for_timeout(900)
        az, el = js("[M('cam_src').data.az[0], M('cam_src').data.el[0]]")
        check("a preset button compiles and runs the redraw JS",
              abs(az + math.pi / 2) < 1e-9 and abs(el) < 1e-9,
              "az=%.4f el=%.4f" % (az, el))
        check("  ... and no JS error came out of it", not errors,
              "; ".join(errors[:2]))

        # ------------------------------------------------------------------
        # THE cross-mirror check: browser u/v vs Python u/v
        # ------------------------------------------------------------------
        got = js("""(() => {
            const s = M('cloud_src').data, c = M('cam_src').data;
            const n = s.x.length, k = Math.max(1, Math.floor(n / 200)), out = [];
            for (let i = 0; i < n; i += k)
                out.push([s.x[i], s.y[i], s.z[i], s.u[i], s.v[i]]);
            return {pts: out, az: c.az[0], el: c.el[0],
                    c: [c.cx[0], c.cy[0], c.cz[0]]};
        })()""")
        want = D3.project([(p[0], p[1], p[2]) for p in got["pts"]],
                          got["az"], got["el"], tuple(got["c"]))
        worst = max(max(abs(w[0] - p[3]), abs(w[1] - p[4]))
                    for w, p in zip(want, got["pts"]))
        # float32 columns at a ~500 cm coordinate carry ~6e-5 cm of rounding.
        check("browser projection == Python projection (the two mirrors agree)",
              worst < 1e-3, "%d points, worst |du|,|dv| = %.2e cm"
              % (len(want), worst))

        # ------------------------------------------------------------------
        # gestures
        # ------------------------------------------------------------------
        def drag(dx, dy, modifier=None, x0=None, y0=None):
            sx, sy = (cx if x0 is None else x0), (cy if y0 is None else y0)
            page.mouse.move(sx, sy)
            if modifier:
                page.keyboard.down(modifier)
            page.mouse.down()
            for f in (0.25, 0.5, 0.75, 1.0):
                page.mouse.move(sx + dx * f, sy + dy * f)
                page.wait_for_timeout(60)
            page.mouse.up()
            if modifier:
                page.keyboard.up(modifier)
            page.wait_for_timeout(500)

        az0 = js("M('cam_src').data.az[0]")
        u0 = js("M('cloud_src').data.u[0]")
        drag(160, 0)
        az1, u1 = js("[M('cam_src').data.az[0], M('cloud_src').data.u[0]]")
        check("a bare drag ROTATES (Pan reaches js_on_event with no pan tool)",
              abs(az1 - az0) > 0.1 and abs(u1 - u0) > 1e-6,
              "az %.3f -> %.3f" % (az0, az1))
        el0 = js("M('cam_src').data.el[0]")
        drag(0, -120)
        check("  ... vertical drag changes elevation and clamps below the pole",
              abs(js("M('cam_src').data.el[0]") - el0) > 0.1
              and abs(js("M('cam_src').data.el[0]")) < math.pi / 2)
        check("  ... and the camera reaches the SERVER on panend, once",
              abs(js("M('cam_src').data.az[0]")
                  - float(page.evaluate(
                      "() => [...Bokeh.documents[0].all_models.values()]"
                      ".filter(m => m.type == 'TextInput' && !m.visible)"
                      "[0].value.split(',')[0]")) ) < 1e-3)

        # shift+drag pans the ranges, it does not rotate
        xs0 = js("M('f3d').x_range.start")
        azp = js("M('cam_src').data.az[0]")
        drag(140, 0, modifier="Shift")
        check("shift+drag PANS the view and leaves the camera alone",
              abs(js("M('f3d').x_range.start") - xs0) > 1.0
              and abs(js("M('cam_src').data.az[0]") - azp) < 1e-9,
              "x_range.start %.1f -> %.1f" % (xs0, js("M('f3d').x_range.start")))

        # the wheel zooms, via the real WheelZoomTool
        span0 = js("M('f3d').x_range.end - M('f3d').x_range.start")
        page.mouse.move(cx, cy)
        for _ in range(4):
            page.mouse.wheel(0, -120)
            page.wait_for_timeout(120)
        page.wait_for_timeout(400)
        span1 = js("M('f3d').x_range.end - M('f3d').x_range.start")
        check("the wheel zooms (WheelZoomTool is the active scroll gesture)",
              span1 < span0 * 0.95, "span %.0f -> %.0f cm" % (span0, span1))

        # ------------------------------------------------------------------
        # Box Select must SUSPEND rotation -- the bug that a guard on
        # toolbar.active_drag would have shipped silently.
        # ------------------------------------------------------------------
        activated = page.evaluate("""() => {
            const doc = Bokeh.documents[0];
            for (const m of doc.all_models.values())
                if (m.type == 'BoxSelectTool') { m.active = true; return true; }
            return false;
        }""")
        page.wait_for_timeout(400)
        check("box select can be activated", activated)
        gest = js("M('f3d').toolbar.gestures.pan.active != null")
        check("  ... and it registers as the live pan gesture", gest)
        azb = js("M('cam_src').data.az[0]")
        # A box over most of the frame: the view is fitted to the reconstruction,
        # so this MUST enclose fitted points or the pick surface is not drawing.
        drag(box["w"] * 0.8, box["h"] * 0.8,
             x0=box["x"] + box["w"] * 0.1, y0=box["y"] + box["h"] * 0.1)
        check("  ... so a drag NO LONGER rotates (the guard works)",
              abs(js("M('cam_src').data.az[0]") - azb) < 1e-9)
        nsel = js("M('pick_src').selected.indices.length")
        nseg = js("new Set(M('pick_src').selected.indices"
                  ".map(i => M('pick_src').data.sid[i])).size")
        check("  ... and the box selected fitted points, resolving to SEGMENTS",
              nsel > 5 and 0 < nseg < nsel,
              "%s point(s) -> %s segment(s)" % (nsel, nseg))
        # ------------------------------------------------------------------
        # round 4 requests 3 and 4, exercised through the same box gesture:
        # the selection is now VISIBLE, and the gesture itself can mark.
        # ------------------------------------------------------------------
        def bigbox():
            drag(box["w"] * 0.8, box["h"] * 0.8,
                 x0=box["x"] + box["w"] * 0.1, y0=box["y"] + box["h"] * 0.1)
            page.wait_for_timeout(700)

        def set_tap(v):
            page.evaluate("(v) => { Bokeh.documents[0]"
                          ".get_model_by_name('tap_action').value = v; }", v)
            page.wait_for_timeout(700)

        nsel_halo = js("M('sel3_src').data.xs3.length")
        check("the selection is DRAWN, not just held (cyan halo)",
              nsel_halo > 0 and nsel_halo == nseg,
              "%s halo polyline(s) for %s selected segment(s)"
              % (nsel_halo, nseg))

        # Round 5: a mark is filed AGAINST a shower, so with no row picked the
        # gesture must be refused rather than dropped into a nameless bucket.
        set_tap("mark IN")
        bigbox()
        check("with no shower picked, a marking box is REFUSED",
              js("M('in3_src').data.xs3.length") == 0
              and "pick a shower" in js("M('save_note').text"),
              js("M('save_note').text")[:70])

        def pick_row(i=0):
            page.evaluate("(i) => { Bokeh.documents[0]"
                          ".get_model_by_name('shower_src').selected.indices"
                          " = [i]; }", i)
            page.wait_for_timeout(900)

        pick_row(0)
        check("  ... and picking a shower row switches to the 3-D tab",
              js("M('view_tabs').active") == 0)
        bigbox()
        n_in = js("M('in3_src').data.xs3.length")
        check("a box in 'mark IN' mode marks on the gesture itself",
              n_in > 0, "%s segment(s) marked IN" % n_in)
        check("  ... and the mark list names the shower it was filed against",
              js("M('shower_src').data.node[0]").__str__()
              in js("M('marks_div').text"),
              js("M('marks_div').text")[-140:])
        check("  ... and it re-arms, so the same segments can be clicked again",
              js("M('pick_src').selected.indices.length") == 0)
        set_tap("mark OUT")
        bigbox()
        check("  ... 'mark OUT' moves them across",
              js("M('out3_src').data.xs3.length") == n_in
              and js("M('in3_src').data.xs3.length") == 0,
              "in %s / out %s" % (js("M('in3_src').data.xs3.length"),
                                  js("M('out3_src').data.xs3.length")))

        set_tap("orbit around it")
        cx0 = js("M('cam_src').data.cx[0]")
        u_before = js("M('cloud_src').data.u[0]")
        bigbox()
        check("'orbit around it' moves the camera centre in the browser",
              abs(js("M('cam_src').data.cx[0]") - cx0) > 1e-6,
              "cx %.1f -> %.1f" % (cx0, js("M('cam_src').data.cx[0]")))
        check("  ... and the whole cloud is REPROJECTED about the new centre",
              abs(js("M('cloud_src').data.u[0]") - u_before) > 1e-6)
        set_tap("select segment(s)")

        page.evaluate("() => { for (const m of Bokeh.documents[0].all_models"
                      ".values()) if (m.type == 'BoxSelectTool') m.active"
                      " = false; }")
        page.wait_for_timeout(300)
        azc = js("M('cam_src').data.az[0]")
        drag(120, 0)
        check("  ... turning it off gives rotation back",
              abs(js("M('cam_src').data.az[0]") - azc) > 0.1)

        # round 4 request 2: the cloud on screen is the candidate, not the readout
        filt = page.evaluate("""() => {
            const d = Bokeh.documents[0].get_model_by_name('cloud_div');
            return d ? d.text : null; }""")
        check("the cloud drawn is the neutrino candidate, and says so",
              filt is not None and "carry the reconstruction" in filt,
              (filt or "")[-150:].replace("<br>", " "))

        # ------------------------------------------------------------------
        # the 2-D projections moved into a lazily-rendered tab -- prove they
        # still lay out, and that the two-panel tap still fills x/y/z.
        # ------------------------------------------------------------------
        page.get_by_text("2-D projections", exact=True).click()
        page.wait_for_timeout(2500)
        # Each figure owns TWO stacked canvases (the plot and its overlay layer)
        # at the same rect, so by_size dedupes by position -- without that, "two
        # panels" is one panel twice, which is exactly how the first version of
        # this test filled x and y but never z.
        # Dedupe on x AND y: the panels are stacked 2-over-1 since round 4, so
        # f_xy and f_xz share a left edge and an x-only dedupe throws one away.
        rects = [r for n in ("f_xy", "f_yz", "f_xz") for r in by_size(n)]
        rects = [r for i, r in enumerate(rects)
                 if all(abs(q["x"] - r["x"]) > 2 or abs(q["y"] - r["y"]) > 2
                        for q in rects[:i])]
        check("the 2-D projections still lay out in their (lazy) tab",
              len(rects) == 3, "%d distinct panels at full size" % len(rects))
        # tap-to-fill lives in the pi0 panel, which is hidden in EM mode.  Set the
        # mode through the model rather than by clicking: the click still has to
        # round-trip to the server and back before the panel exists, and waiting
        # on the widget is what we are trying to avoid flaking on.
        page.evaluate("""() => {
            for (const m of Bokeh.documents[0].all_models.values()) {
                if (m.type == 'RadioButtonGroup' && m.labels
                    && m.labels.includes('pi0')) { m.active = 1; }
                if (m.type == 'Toggle' && m.label
                    && m.label.indexOf('tap fills') >= 0) { m.active = true; }
            }
        }""")
        page.wait_for_timeout(2500)
        tapon = page.evaluate("""() => {
            for (const m of Bokeh.documents[0].all_models.values())
                if (m.type == 'Toggle' && m.label
                    && m.label.indexOf('tap fills') >= 0) return m.active;
            return null;
        }""")
        check("  ... tap-to-fill can be armed", tapon is True, str(tapon))
        for r in rects[:2]:
            page.mouse.click(r["x"] + r["w"] * 0.55, r["y"] + r["h"] * 0.45)
            page.wait_for_timeout(500)
        vals = page.evaluate("""() => [...Bokeh.documents[0].all_models.values()]
            .filter(m => m.type == 'TextInput' && ['x','y','z'].includes(m.title))
            .map(m => m.value)""")
        check("  ... and a tap in two panels still pins a 3-D point",
              len(vals) == 3 and all(v not in ("", None) for v in vals),
              str(vals))

        page.get_by_text("3-D", exact=True).first.click()
        page.wait_for_timeout(1200)

        # ------------------------------------------------------------------
        # round 5: the tables, the acceptance plot and the 3-D view are one
        # brushed set, and whole showers can be dimmed out of the way.
        # ------------------------------------------------------------------
        page.evaluate("""() => { const d = Bokeh.documents[0];
            for (const m of d.all_models.values())
                if (m.type == 'RadioButtonGroup' && m.labels
                    && m.labels.includes('EM shower')) m.active = 0; }""")
        page.wait_for_timeout(1200)
        page.evaluate("() => { Bokeh.documents[0]"
                      ".get_model_by_name('shower_src').selected.indices"
                      " = [0]; }")
        page.wait_for_timeout(1500)
        # Pick a candidate row that is NOT already selected anywhere.
        sid = page.evaluate("() => Bokeh.documents[0]"
                            ".get_model_by_name('cand_src').data.sid[2]")
        page.evaluate("() => { Bokeh.documents[0]"
                      ".get_model_by_name('cand_src').selected.indices = [2]; }")
        page.wait_for_timeout(1500)
        plot_sel = page.evaluate("""() => { const d = Bokeh.documents[0];
            const s = d.get_model_by_name('cand_pt_src');
            return s.selected.indices.map(i => s.data.sid[i]); }""")
        pick_sel = page.evaluate("""() => { const d = Bokeh.documents[0];
            const s = d.get_model_by_name('pick_src');
            return [...new Set(s.selected.indices.map(i => s.data.sid[i]))]; }""")
        check("a candidate-table click brushes the plot and the 3-D view",
              plot_sel == [sid] and pick_sel == [sid],
              "row %s -> plot %s, 3-D %s" % (sid, plot_sel, pick_sel))
        check("  ... and the cyan halo follows, once",
              js("M('sel3_src').data.xs3.length") == 1,
              str(js("M('sel3_src').data.xs3.length")))

        opts = page.evaluate("() => Bokeh.documents[0]"
                             ".get_model_by_name('excl_choice').options")
        page.evaluate("(o) => { Bokeh.documents[0]"
                      ".get_model_by_name('excl_choice').value = [o]; }",
                      opts[1])
        page.wait_for_timeout(1800)
        faded = page.evaluate("() => { const s = Bokeh.documents[0]"
                              ".get_model_by_name('seg3_src');"
                              " return s.data.a.filter(v => v < 0.1).length; }")
        check("dimming a shower fades exactly its segments in the live 3-D view",
              faded > 0, "%s faded polyline(s)" % faded)
        page.evaluate("() => { Bokeh.documents[0]"
                      ".get_model_by_name('excl_choice').value = []; }")
        page.wait_for_timeout(1200)
        check("  ... and clearing it brings them back",
              page.evaluate("() => { const s = Bokeh.documents[0]"
                            ".get_model_by_name('seg3_src');"
                            " return s.data.a.filter(v => v < 0.1).length; }") == 0)

        cols = page.evaluate("""() => { const d = Bokeh.documents[0];
            const seg = d.get_model_by_name('seg3_src').data;
            const m = {};
            for (let i = 0; i < seg.sid.length; i++) {
                const o = seg.owner[i];
                if (o === '-' ) continue;
                (m[o] = m[o] || new Set()).add(seg.c[i]);
            }
            return Object.entries(m).map(([k, v]) => [k, v.size]); }""")
        check("in the live view each shower is drawn in one colour",
              cols and all(n == 1 for _, n in cols),
              str([c for c in cols if c[1] != 1]))

        # ---- round 6: one of the added events, end to end -------------------
        # The static test proves the round/idx binding arithmetically.  This
        # proves the zip built this round actually opens in the RUNNING app:
        # the four additions are the first rows whose Bee set was never
        # uploaded, so they are the first to exercise the bee_round-without-
        # bee_url path all the way to a rendered cloud.
        page.evaluate("() => { Bokeh.documents[0]"
                      ".get_model_by_name('event_select').value = 'evt169626'; }")
        page.wait_for_timeout(6000)
        npts = js("M('cloud_src').data.x.length")
        nseg = js("M('seg3_src').data.xs3.length")
        check("a round-6 addition renders its own cloud in the live app",
              npts > 500 and nseg > 0,
              "%s cloud points, %s polylines" % (npts, nseg))
        banner_txt = page.evaluate("() => Bokeh.documents[0]"
                                   ".get_model_by_name('banner').text")
        check("  ... and the banner says the set is built, not missing",
              "not uploaded" in banner_txt and "no Bee set" not in banner_txt)
        note_txt = page.evaluate("() => Bokeh.documents[0]"
                                  ".get_model_by_name('scan_note_div').text")
        check("  ... with the owner's hint for THIS event on screen",
              "567 MeV gamma" in note_txt, note_txt[:80])

        check("no JS errors over the whole session", not errors,
              "; ".join(errors[:3]))
        browser.close()
finally:
    stop_server()

print()
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)

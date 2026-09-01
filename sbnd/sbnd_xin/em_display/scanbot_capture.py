#!/usr/bin/env python3
"""Phase A of the agent hand scan (doc pr/116): capture, no judgement.

Drives a RUNNING em_display server with headless chromium, rotates the 3-D view
about the neutrino vertex, and writes PNGs plus a state.json for one event.  It
makes no judgement and it NEVER clicks Save -- recording is scanbot_record.py,
deliberately a second file so a re-judgement does not force a re-capture.

    ./scanbot_capture.py --event evt400504 [--port 5019]

"Never saves" is not the same as "touches nothing".  Be precise about it, because
this runs against a LIVE server: selecting an event, a shower or a camera is a
server-side state change on this session, and with --pio-pairs the pi0 block also
sets mode_group and assigns the two gamma slots, which is the only way to read the
two mass conventions back out.  All of it is confined to this browser session,
which is new on every invocation, and nothing reaches disk -- Save is never
clicked and no label is written.

Point it ONLY at your own server with your own --scan-tag: em_display writes
labels into whatever tag its server was launched with, and emscan-0827 /
emscan-0828-beam141 are scientific records (CLAUDE.md M13).  --expect-tag makes
that a refusal rather than a convention.

The idioms are lifted from selftest_em3d_browser.py:120-240 -- the shadow-root
canvas walk, `js()`, `setm()`, the real-mouse `drag()` -- which is the file that
proved they work against this app.  This is a fork, not an import: that one is a
self-test with its own server lifecycle and its own exit contract.

Two things learned the hard way and encoded here:

* **The camera is written, not dragged.**  A drag is 0.0075 rad/px with the
  elevation clamped at the pole (em3d.py:524-530), so a fixed sequence of drags
  ratchets: by the second shower every view was top-down and two shots were
  byte-identical.  Angles are set through `cam_src` instead -- the same route the
  preset buttons take (em3d.py:116-122) -- so every event is shot from the same
  camera.  One real drag is still performed, as the check that the pixels move.
* **A clip must be scrolled into view first.**  `getBoundingClientRect` is
  viewport-relative; the acceptance plot sits below the fold and its first
  screenshot came back silently truncated to 200 px of its 330.
"""
import argparse, json, math, os, re, sys, time

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright not importable -- see selftest_em3d_browser.py:51")

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

TAG_RE = re.compile(r"<[^>]+>")


def detag(html):
    if not html:
        return ""
    s = html
    for a, b in (("<br>", "\n"), ("<br/>", "\n"), ("</div>", "\n"),
                 ("</p>", "\n"), ("</tr>", "\n"), ("</li>", "\n")):
        s = s.replace(a, b)
    s = TAG_RE.sub(" ", s)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&mdash;", "--"), ("&sigma;", "sigma"), ("&Sigma;", "Sum"),
                 ("&plusmn;", "+-"), ("&deg;", "deg"), ("&rarr;", "->"),
                 ("&gamma;", "gamma"), ("&pi;", "pi"), ("&nu;", "nu"),
                 ("&theta;", "theta"), ("&chi;", "chi"), ("&times;", "x")):
        s = s.replace(a, b)
    s = re.sub(r"[ \t]+", " ", s)
    return "\n".join(ln.strip() for ln in s.split("\n") if ln.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--port", type=int, default=5019)
    ap.add_argument("--out", default="/home/xqian/tmp/emscan-agent5")
    ap.add_argument("--expect-tag", default="emscan-0828-agent5",
                    help="refuse to run unless the served scan tag is this one;"
                         " pass '' to skip the check")
    ap.add_argument("--panel", default="1100")
    ap.add_argument("--min-kb", type=float, default=20.0)
    ap.add_argument("--max-showers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--pio-pairs", default=None,
                    help="extra gamma pairings to price, e.g. '12119:95114,24015:95114'."
                         " The display computes both mass conventions; nothing is saved.")
    opts = ap.parse_args()

    url = "http://localhost:%d/em_display_viewer" % opts.port
    odir = os.path.join(opts.out, opts.event)
    os.makedirs(odir, exist_ok=True)
    shots, log = [], []

    def note(s):
        print(s, flush=True)
        log.append(s)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 2000, "height": 1400},
                                device_scale_factor=opts.scale)
        errors = []
        page.on("console", lambda m: errors.append("%s: %s" % (m.type, m.text))
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
        page.goto(url, wait_until="networkidle")
        page.wait_for_function(
            "() => window.Bokeh && Bokeh.documents && Bokeh.documents.length > 0",
            timeout=60000)
        page.wait_for_timeout(4000)

        def js(expr):
            return page.evaluate("() => { const doc = Bokeh.documents[0];"
                                 " const M = (n) => doc.get_model_by_name(n);"
                                 " return (%s); }" % expr)

        def setm(name, prop, val, wait=800):
            page.evaluate("() => { Bokeh.documents[0].get_model_by_name('%s')"
                          ".%s = %s; }" % (name, prop, json.dumps(val)))
            page.wait_for_timeout(wait)

        # Several view controls carry no name= (em_display_viewer.py:565-618).
        # Reach them through all_models by their own title / labels rather than
        # by DOM position, which the round-4 two-column layout would break.
        def set_titled(title, prop, val, wait=900):
            ok = page.evaluate(
                """(a) => { for (const m of Bokeh.documents[0].all_models.values())
                     if (m.title === a.t) { m[a.p] = a.v; return true; }
                   return false; }""", {"t": title, "p": prop, "v": val})
            page.wait_for_timeout(wait)
            return ok

        def set_radio(first_label, active, wait=900):
            ok = page.evaluate(
                """(a) => { for (const m of Bokeh.documents[0].all_models.values())
                     if (m.labels && m.labels[0] === a.l) { m.active = a.i; return true; }
                   return false; }""", {"l": first_label, "i": active})
            page.wait_for_timeout(wait)
            return ok

        # Write the camera the way the preset buttons do (em3d.py:116-122):
        # mutate cam_src and hand it a fresh object so js_on_change fires
        # JS_REDRAW.  Route A of the two in the app -- it does not update the
        # PYTHON-side state["cam"], so any later Python push (a shower-row click)
        # overrides it.  That is fine and deliberate: each block below either
        # owns the camera or re-reads it after a push.
        def cam_set(az_deg=None, el_deg=None, centre=None, R=None, wait=700):
            a = dict(az=None if az_deg is None else math.radians(az_deg),
                     el=None if el_deg is None else math.radians(el_deg),
                     c=list(centre) if centre else None, R=R)
            page.evaluate(
                """(a) => {
                    const doc = Bokeh.documents[0];
                    const c = doc.get_model_by_name('cam_src');
                    const f = doc.get_model_by_name('f3d');
                    const d = Object.assign({}, c.data);
                    if (a.az !== null) d.az = [a.az];
                    if (a.el !== null) d.el = [a.el];
                    if (a.c !== null) { d.cx=[a.c[0]]; d.cy=[a.c[1]]; d.cz=[a.c[2]]; }
                    c.data = d;
                    if (a.R !== null) {
                        f.x_range.start = -a.R; f.x_range.end = a.R;
                        f.y_range.start = -a.R; f.y_range.end = a.R;
                    }
                }""", a)
            page.wait_for_timeout(wait)

        def canvas(model, tol=8):
            try:
                return page.evaluate(
                    "() => { const m = Bokeh.documents[0].get_model_by_name('%s');"
                    " const r = %s.filter(r => Math.abs(r.w - m.width) < %d"
                    " && Math.abs(r.h - m.height) < %d)"
                    " .filter((r,i,a) => a.findIndex(q => Math.abs(q.x-r.x) < 2"
                    "   && Math.abs(q.y-r.y) < 2) === i);"
                    " return r.length ? r[0] : null; }" % (model, RECTS, tol, tol))
            except Exception:
                return None

        VH = 1400

        def shot(tag, model="f3d"):
            page.mouse.move(3, 3)            # keep hover tooltips out of frame
            page.wait_for_timeout(120)
            path = os.path.join(odir, "%s.png" % tag)
            box = canvas(model)
            if box is not None and (box["y"] < 0 or box["y"] + box["h"] > VH):
                # getBoundingClientRect is viewport-relative: a canvas below the
                # fold clips SILENTLY TRUNCATED, not to an error.
                page.evaluate("(dy) => window.scrollBy(0, dy)", box["y"] - 60)
                page.wait_for_timeout(500)
                box = canvas(model)
            if box is None:
                page.screenshot(path=path, full_page=(model == "__page__"))
            else:
                page.screenshot(path=path, clip={"x": box["x"], "y": box["y"],
                                                 "width": box["w"], "height": box["h"]})
            cam = js("[M('cam_src').data.az[0], M('cam_src').data.el[0],"
                     " M('cam_src').data.cx[0], M('cam_src').data.cy[0],"
                     " M('cam_src').data.cz[0], M('f3d').x_range.start,"
                     " M('f3d').x_range.end]")
            rec = dict(tag=tag, file=os.path.basename(path), model=model,
                       az_deg=round(math.degrees(cam[0]), 2),
                       el_deg=round(math.degrees(cam[1]), 2),
                       centre=[round(c, 2) for c in cam[2:5]],
                       half_span_cm=round((cam[6] - cam[5]) / 2.0, 2),
                       bytes=os.path.getsize(path))
            shots.append(rec)
            note("  %-24s az %7.2f el %6.2f  centre (%.0f,%.0f,%.0f)  +-%.0f cm  %dkB"
                 % (tag, rec["az_deg"], rec["el_deg"], rec["centre"][0],
                    rec["centre"][1], rec["centre"][2], rec["half_span_cm"],
                    rec["bytes"] // 1024))
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(200)
            return rec

        def drag(dx, dy, modifier=None):
            box = canvas("f3d")
            sx, sy = box["x"] + box["w"] / 2.0, box["y"] + box["h"] / 2.0
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
            page.wait_for_timeout(600)

        def preset(name):
            page.get_by_role("button", name=name, exact=True).click()
            page.wait_for_timeout(900)

        def data(name, cols=None):
            d = js("M('%s') ? M('%s').data : null" % (name, name))
            if d and cols:
                d = {k: v for k, v in d.items() if k in cols}
            return d

        def divtext(name):
            try:
                return detag(js("M('%s') ? M('%s').text : ''" % (name, name)))
            except Exception:
                return ""

        # The server decides the tag, not this script.  Read it back off the
        # page rather than trusting --port: a mistyped port would put this
        # session on the owner's live scan (M13).
        status = js("M('scan_status') ? M('scan_status').text : ''") or ""
        if opts.expect_tag and opts.expect_tag not in status:
            sys.exit("port %d is NOT serving tag %r (status: %s) -- refusing"
                     % (opts.port, opts.expect_tag, detag(status)[:200]))

        # ---------------------------------------------------------------- load
        setm("event_select", "value", opts.event, wait=7000)
        got = js("M('event_select').value")
        if got != opts.event:
            sys.exit("event_select did not take: asked %s, got %s" % (opts.event, got))
        set_titled("3-D panel size", "value", opts.panel, wait=2500)
        note("event %s | panel %s px | %s" % (opts.event, js("M('f3d').width"),
                                              divtext("banner")[:160]))
        note("scan status: %s" % divtext("scan_status")[:120])

        shot("page", model="__page__")

        showers = data("shower_src")
        seg3 = data("seg3_src")
        mv = data("mainvtx3_src")
        vx, vy, vz = mv["x"][0], mv["y"][0], mv["z"][0]

        # Event scale: how far the reconstruction's own segments reach from the
        # neutrino vertex.  `frame the reco` uses the bounding sphere of
        # EVERYTHING including far vertices, which on evt400504 was R 237 cm for
        # an event 20 cm across -- the whole event was 100 px of 2200.
        rr = []
        for xs, ys, zs in zip(seg3["xs3"], seg3["ys3"], seg3["zs3"]):
            for x, y, z in zip(xs, ys, zs):
                rr.append(math.sqrt((x - vx) ** 2 + (y - vy) ** 2 + (z - vz) ** 2))
        rr.sort()
        r95 = rr[int(0.95 * (len(rr) - 1))] if rr else 50.0
        r_evt = max(25.0, min(400.0, 1.25 * r95))
        note("event scale: %d segment points, r95 = %.1f cm from the nu vertex"
             " -> half-span %.1f cm" % (len(rr), r95, r_evt))

        row_of = {int(v): i for i, v in enumerate(showers.get("node", []))}
        n = len(showers.get("node", []))
        order = sorted(range(n), key=lambda i: -float(showers["E"][i]))
        scope = [i for i in order
                 if float(showers["kb"][i]) >= opts.min_kb
                 or int(showers["pio"][i]) >= 0][:opts.max_showers]
        skipped = [int(showers["node"][i]) for i in order if i not in scope]
        note("showers: %d total, %d in scope (kine_best >= %.0f MeV or pio_id >= 0,"
             " top %d by kine_charge); skipped %s"
             % (n, len(scope), opts.min_kb, opts.max_showers, skipped or "none"))

        # ---------------------------------------------- context: the whole reco
        set_radio("frame the reco", 0)
        page.get_by_role("button", name="refit", exact=True).click()
        page.wait_for_timeout(1000)
        preset("iso"); shot("ov-iso")
        preset("x-z"); shot("ov-xz")

        # ------------------------------- the sweep about the neutrino vertex
        # This is the ask, literally: the camera centre IS the nu vertex and only
        # the angles change, so every frame shows the same point from a new side.
        SWEEP = ((-55, 20), (5, 20), (65, 20), (125, 20), (-55, 60), (-55, -25))
        for az, el in SWEEP:
            cam_set(az, el, centre=(vx, vy, vz), R=r_evt)
            shot("vtx-az%+04d-el%+03d" % (az, el))

        # --- the check that the PIXELS moved, not just the readout ------------
        # JS_REDRAW reprojects only registered sources (em3d.py _LINE3 / _PT_SRC);
        # one outside those registries would freeze at the camera it was last
        # filled from while cam_src read perfectly correct.
        cam_set(-55, 20, centre=(vx, vy, vz), R=r_evt)
        az0 = js("M('cam_src').data.az[0]")
        u0 = js("M('seg3_src').data.xs[0][0]")
        c0 = js("M('cloud_src') && M('cloud_src').data.u.length"
                " ? M('cloud_src').data.u[0] : null")
        drag(100, 0)
        az1 = js("M('cam_src').data.az[0]")
        u1 = js("M('seg3_src').data.xs[0][0]")
        c1 = js("M('cloud_src') && M('cloud_src').data.u.length"
                " ? M('cloud_src').data.u[0] : null")
        drag_check = dict(
            az_rad_before=az0, az_rad_after=az1, d_az_rad=az1 - az0,
            expect_rad=100 * 0.0075, seg_u_before=u0, seg_u_after=u1,
            cloud_u_before=c0, cloud_u_after=c1,
            segments_redrew=abs(u1 - u0) > 1e-6,
            cloud_redrew=(c0 is not None and abs(c1 - c0) > 1e-6))
        note("drag check: 100 px -> d_az %.4f rad (expect %.4f); segments moved %s;"
             " cloud moved %s" % (drag_check["d_az_rad"], drag_check["expect_rad"],
                                  drag_check["segments_redrew"],
                                  drag_check["cloud_redrew"]))

        # --------------------------------------------------------- per shower
        set_radio("frame the reco", 2)          # frame the shower
        per = []
        for i in scope:
            node = int(showers["node"][i])
            setm("shower_src", "selected.indices", [i], wait=2000)
            base = js("[M('cam_src').data.cx[0], M('cam_src').data.cy[0],"
                      " M('cam_src').data.cz[0], M('f3d').x_range.end]")
            ctr, rad = base[:3], base[3]
            for az, el in ((-55, 20), (15, 20), (85, 20), (-55, 65)):
                cam_set(az, el, centre=ctr, R=rad)
                shot("shw%d-az%+04d-el%+03d" % (node, az, el))
            shot("shw%d-acc" % node, model="acc")
            per.append(dict(
                index=i, node=node, colour=showers["color"][i],
                pdg=int(showers["pdg"][i]), nseg=int(showers["nseg"][i]),
                joined=showers["joined"][i], kine_charge=float(showers["E"][i]),
                kine_best=float(showers["kb"][i]), length=float(showers["length"][i]),
                conn=int(showers["conn"][i]), pio_id=int(showers["pio"][i]),
                frame_centre=[round(c, 2) for c in ctr], frame_half_span=round(rad, 2),
                candidates=data("cand_src"), cmp_div=divtext("cmp_div"),
                marks_div=divtext("marks_div")))
            note("  shower %-7d pdg %-5s nseg %-3d kine_charge %8.1f kine_best %8.1f"
                 " len %6.1f conn %s pio_id %s"
                 % (node, per[-1]["pdg"], per[-1]["nseg"], per[-1]["kine_charge"],
                    per[-1]["kine_best"], per[-1]["length"], per[-1]["conn"],
                    per[-1]["pio_id"]))

        # ------------------------------------------------------------- pi0 mode
        # kine_div is EMPTY outside pi0 mode (refresh_kine returns early at
        # em_display_viewer.py:3013), so the kine_pio_* block and the two mass
        # conventions are only readable here.  Nothing is saved.
        pio = {}
        pair = [i for i in scope if int(showers["pio"][i]) >= 0][:2]
        if len(pair) < 2:
            pair = [i for i in scope if int(showers["pdg"][i]) == 11][:2]
        setm("mode_group", "active", 1, wait=1800)
        if len(pair) == 2:
            for slot, i in zip((1, 2), pair):
                setm("shower_src", "selected.indices", [i], wait=1200)
                page.get_by_role(
                    "button", name="selected shower -> gamma %d" % slot,
                    exact=True).click()
                page.wait_for_timeout(1500)
            cam_set(-55, 20, centre=(vx, vy, vz), R=r_evt)
            shot("pio-az-055-el+020")
            cam_set(35, 20); shot("pio-az+035-el+020")
            pio["slots"] = [int(showers["node"][i]) for i in pair]
        # Extra pairings, priced by the display's own formula rather than by
        # hand: the reconstruction's accepted pair is only one of several, and
        # `kine_pio_*` is a third thing again (pr/114 SS6.2).
        for spec in (opts.pio_pairs or "").split(","):
            if ":" not in spec:
                continue
            a, b = [int(x) for x in spec.split(":")]
            if a not in row_of or b not in row_of:
                note("  pairing %s: not both showers are in this event" % spec)
                continue
            for slot, nd in ((1, a), (2, b)):
                setm("shower_src", "selected.indices", [row_of[nd]], wait=1400)
                page.get_by_role("button", name="selected shower -> gamma %d" % slot,
                                 exact=True).click()
                page.wait_for_timeout(1600)
            pio.setdefault("alt_pairings", {})[spec] = divtext("kine_div")
            cam_set(-55, 20, centre=(vx, vy, vz), R=r_evt)
            shot("pio-alt-%d-%d" % (a, b))
            note("  priced alternative pairing %d + %d" % (a, b))
        pio["kine_div"] = divtext("kine_div")
        pio["pio_cand_div"] = divtext("pio_cand_div")
        pio["marks_div"] = divtext("marks_div")
        setm("mode_group", "active", 0, wait=1200)
        note("pi0 block: slots %s; kine_div %d chars"
             % (pio.get("slots"), len(pio["kine_div"])))

        state = dict(
            event=opts.event, url=url,
            captured_utc=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            panel_px=js("M('f3d').width"), device_scale_factor=opts.scale,
            scope_rule=dict(min_kine_best=opts.min_kb, max_showers=opts.max_showers,
                            in_scope=[int(showers["node"][i]) for i in scope],
                            skipped=skipped),
            event_half_span_cm=r_evt, r95_cm=r95,
            drag_check=drag_check,
            banner=divtext("banner"), scan_status=divtext("scan_status"),
            cloud=divtext("cloud_div"), emstart=divtext("emstart_div"),
            main_vertex={k: (v[0] if isinstance(v, list) and v else v)
                         for k, v in mv.items()},
            vertices=data("vtx3_src"), shower_table=showers,
            segments=seg3, showers=per, pio=pio,
            shots=shots, console_errors=errors, log=log)
        with open(os.path.join(odir, "state.json"), "w") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
        note("wrote %s/state.json  (%d shots, %d showers, %d console errors)"
             % (odir, len(shots), len(per), len(errors)))
        if errors:
            note("CONSOLE: %s" % "; ".join(errors[:3]))
        browser.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase C of the agent hand scan (doc pr/116): replay a judgement into the app.

Reads a decisions JSON written by hand after looking at the Phase-A captures, and
drives the display to record it exactly as a person would: select a shower, mark
its segments, assign the pi0 gamma slots, set the verdict radio, type the note,
press Save.  Nothing here judges anything.

    ./scanbot_record.py --decisions decisions-agent5.json [--port 5019] [--dry-run]

Separate from scanbot_capture.py on purpose: a re-judgement must not force a
re-capture, and a re-capture must not overwrite a judgement.

Decisions schema, one object per event:

    {"event": "evt400504",
     "verdict_shower": 63024,          # em.verdict is stored for ONE shower --
     "verdict": "correct",             #   the one selected at save time
     "confidence": "likely",           #   (em_display_viewer.py:4160-4218)
     "note": "[agent] ...",
     "event_flags": [],                # subset of ["no_vertex_ncpi0"]
     "marks": {"62014": {"21003": "out"}},        # {shower: {segment: in|out|?}}
     "pio": {"g1": 63024, "g2": 62014, "store": true}}

The verdict and confidence strings must be VERBATIM from EM_VERDICTS / CONF:
`load_label` restores them with `.index()` behind an `in` guard, so a typo does
not raise -- it silently restores as unset (em_display_viewer.py:3930, :4049).

Refuses to run against a port whose scan tag is not the one you named, so a
mis-typed --port cannot write into emscan-0827 or emscan-0828-beam141 (M13).
"""
import argparse, json, os, sys, time

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright not importable")

EM_VERDICTS = ["correct", "over-clustered", "under-clustered", "both",
               "vertex-bad (undecidable)", "not an EM shower",
               "is an EM shower (reco PID wrong)"]
CONF = ["certain", "likely", "unclear"]
EVENT_FLAG_KEYS = ["no_vertex_ncpi0"]
MARK_BTN = {"in": "mark IN", "out": "mark OUT", "?": "mark ?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--port", type=int, default=5019)
    ap.add_argument("--expect-tag", default="emscan-0828-agent5")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--only", default=None, help="record just this event")
    ap.add_argument("--dry-run", action="store_true")
    opts = ap.parse_args()

    decisions = json.load(open(opts.decisions))
    if isinstance(decisions, dict):
        decisions = [decisions]
    if opts.only:
        decisions = [d for d in decisions if d["event"] == opts.only]
    if not decisions:
        sys.exit("no decisions selected")

    for d in decisions:                      # fail before touching the browser
        if d["verdict"] not in EM_VERDICTS:
            sys.exit("event %s: verdict %r is not in EM_VERDICTS" % (d["event"], d["verdict"]))
        if d.get("confidence") and d["confidence"] not in CONF:
            sys.exit("event %s: confidence %r is not in CONF" % (d["event"], d["confidence"]))
        for f in d.get("event_flags", []):
            if f not in EVENT_FLAG_KEYS:
                sys.exit("event %s: unknown event flag %r" % (d["event"], f))

    ldir = opts.labels_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "em_labels", opts.expect_tag)
    url = "http://localhost:%d/em_display_viewer" % opts.port
    ok_all = True

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 2000, "height": 1400})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
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

        def set_titled(title, prop, val, wait=800):
            ok = page.evaluate(
                """(a) => { for (const m of Bokeh.documents[0].all_models.values())
                     if (m.title === a.t) { m[a.p] = a.v; return true; }
                   return false; }""", {"t": title, "p": prop, "v": val})
            page.wait_for_timeout(wait)
            return ok

        def set_radio(first_label, active, wait=800):
            ok = page.evaluate(
                """(a) => { for (const m of Bokeh.documents[0].all_models.values())
                     if (m.labels && m.labels[0] === a.l) { m.active = a.i; return true; }
                   return false; }""", {"l": first_label, "i": active})
            page.wait_for_timeout(wait)
            return ok

        def click(label):
            page.get_by_role("button", name=label, exact=True).click()
            page.wait_for_timeout(1200)

        # The server decides the tag, not this script.  Read it back off the
        # page rather than trusting --port: one stray Save into emscan-0827 or
        # emscan-0828-beam141 would corrupt a scientific record (M13).
        status = js("M('scan_status').text") or ""
        if opts.expect_tag not in status:
            sys.exit("port %d is NOT serving tag %r (status: %s) -- refusing"
                     % (opts.port, opts.expect_tag, status[:200]))
        print("port %d serves tag %s -- ok" % (opts.port, opts.expect_tag))

        for d in decisions:
            ev = d["event"]
            print("\n=== %s ===" % ev)
            setm("event_select", "value", ev, wait=7000)
            if js("M('event_select').value") != ev:
                sys.exit("event_select did not take for %s" % ev)
            nodes = [int(x) for x in js("M('shower_src').data.node")]
            row = {n: i for i, n in enumerate(nodes)}

            for snode, segs in sorted(d.get("marks", {}).items(),
                                      key=lambda kv: int(kv[0])):
                snode = int(snode)
                if snode not in row:
                    sys.exit("%s: shower %d is not in this event" % (ev, snode))
                setm("shower_src", "selected.indices", [row[snode]], wait=1800)
                sids = [str(x) for x in js("M('cand_src').data.sid")]
                for kind in ("in", "out", "?"):
                    want = [s for s, k in segs.items() if k == kind]
                    idx = [j for j, s in enumerate(sids) if s in want]
                    missing = [s for s in want if s not in sids]
                    if missing:
                        sys.exit("%s shower %d: segment(s) %s not in the candidate"
                                 " table" % (ev, snode, missing))
                    if not idx:
                        continue
                    # A mark clears the selection so `toggle` can re-fire on the
                    # same segment, so each kind needs its own fresh selection.
                    setm("cand_src", "selected.indices", idx, wait=700)
                    if not opts.dry_run:
                        click(MARK_BTN[kind])
                    print("  shower %-7d %-3s %s" % (snode, kind, sorted(want)))

            pio = d.get("pio")
            if pio and not opts.dry_run:
                setm("mode_group", "active", 1, wait=1800)
                for slot in (1, 2):
                    n = int(pio["g%d" % slot])
                    setm("shower_src", "selected.indices", [row[n]], wait=1400)
                    click("selected shower -> gamma %d" % slot)
                if pio.get("vertex_mode") is not None:
                    setm("vtx_mode_group", "active", int(pio["vertex_mode"]), wait=1200)
                if pio.get("store", True):
                    click("store this pairing")
                print("  pi0: gamma1 %s, gamma2 %s, stored %s"
                      % (pio["g1"], pio["g2"], pio.get("store", True)))
                setm("mode_group", "active", 0, wait=1400)

            flags = d.get("event_flags", [])
            setm("event_flag_group", "active",
                 [EVENT_FLAG_KEYS.index(f) for f in flags], wait=700)
            set_titled("note (optional)", "value", d.get("note", ""))

            # The verdict is stored for the shower selected at SAVE time, so this
            # selection is the last one made.
            vn = int(d["verdict_shower"])
            if vn not in row:
                sys.exit("%s: verdict shower %d is not in this event" % (ev, vn))
            setm("shower_src", "selected.indices", [row[vn]], wait=1800)
            set_radio("correct", EM_VERDICTS.index(d["verdict"]))
            if d.get("confidence"):
                set_radio("certain", CONF.index(d["confidence"]))

            if opts.dry_run:
                print("  DRY RUN -- not saving")
                continue
            click("Save event label")
            page.wait_for_timeout(1500)
            # write_allowed() refuses into a tag that already holds labels unless
            # --scan-tag was explicit, and a refusal is a Div message, NOT an
            # exception -- a scripted click would swallow it in silence.
            msg = js("M('save_note').text") or ""
            path = os.path.join(ldir, "labels-%s.json" % ev)
            on_disk = os.path.exists(path)
            good = on_disk and ("refus" not in msg.lower())
            ok_all = ok_all and good
            print("  save: %s | on disk: %s | %s"
                  % ("OK" if good else "FAILED", on_disk,
                     " ".join(msg.split())[:220]))
            if not good:
                sys.exit("save did not land for %s -- stopping before the rest" % ev)

        browser.close()
    print("\n%s" % ("all saves landed" if ok_all else "SOME SAVES FAILED"))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

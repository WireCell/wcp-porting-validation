#!/usr/bin/env python3
"""Regression test for the nusel_scan_viewer bundle table repainting (doc 58).

Bokeh 3.9's DataTable repaints only on its CDSView's change signal, and the view
suppresses that signal whenever the recomputed indices compare equal -- which is
every navigation that leaves the visible row COUNT unchanged, and (worse) every
step in or out of an EMPTY table, since an empty data dict makes the client read
get_length() as null and fall back to size 1.  The bundle table then showed the
PREVIOUS event's rows.  Four cases, run headless against a live viewer:

  1. in-beam-only  evt287517 -> evt287759 -> evt287825   (1 -> 0 -> 1 rows)
  2. all bundles   evt286065 -> evt286197                (9 -> 9 rows)
  3. a scan-label click on the focused row               (n -> n rows)
  4. every event in the scan, table vs its own nusel TSV

Cases 1-3 name MCP2025C events and expect the 30-event d56bw manifest; case 4
adapts to whatever the server was given.  Serve a scratch tag, NOT a live
hand-scan tag -- case 3 writes a label (M13):

  ./serve_nusel_scan.sh 5099 --tag tblrefresh --charge-src pr <work roots> &
  python3 test_table_refresh.py http://localhost:5099/nusel_scan_viewer [tag]

Needs playwright + its chromium (~/.cache/ms-playwright).  Exit 0 = all pass.
"""
import os
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5099/nusel_scan_viewer"
TAG = sys.argv[2] if len(sys.argv) > 2 else "d56bw"


def rows(page):
    """The DataTable's rendered rows, as seen in the DOM (shadow-DOM pierced).

    SlickGrid renders rows out of document order once it has scrolled, so key
    each row by its own '#' cell (column 0) rather than DOM position.
    """
    out = []
    for r in page.query_selector_all(".slick-row"):
        cells = [c.inner_text().strip() for c in r.query_selector_all(".slick-cell")]
        if cells:
            out.append(cells)
    out.sort(key=lambda c: int(c[0]) if c[0].isdigit() else -1)
    return out


def brief(rs):
    """One line per row: # grp t(us) beam main clusters ... scan."""
    if not rs:
        return ["<no rows>"]
    # cells: row grp t_us pe beam main clusters npts len verdicts stmfit prev
    #        auto scan cmt   (index_position=None, so no index column)
    return [" | ".join(c[:8]) + "  scan=" + repr(c[13] if len(c) > 13 else "?")
            for c in rs]


def show(tag, page):
    rs = rows(page)
    print(f"--- {tag}: {len(rs)} row(s)")
    for line in brief(rs):
        print("    " + line)
    return rs


def wait_loaded(page, label):
    """The status Div reads 'Loaded <b>evtNNN</b>: ...' once Python has the event.

    Bokeh 3.9 renders widgets into shadow roots, so body.innerText cannot see
    this -- playwright's selector engine pierces open shadow DOM, so match the
    <b> directly.
    """
    page.wait_for_selector(f"b:text-is('{label}')", timeout=60000)
    page.wait_for_timeout(1500)


def goto_event(page, label):
    page.select_option("select >> nth=0", label)
    wait_loaded(page, label)


def next_event(page, expect):
    page.click("button:has-text('next evt >')")
    wait_loaded(page, expect)


SB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sbnd_xin/


def expected_rows(tag=TAG):
    """{label: (n_rows, first_flash_time_us)} straight from the nusel TSVs --
    the viewer keeps the TSV's row order, so row 0's t(us) identifies the event."""
    import glob
    out = {}
    for t in glob.glob(f"{SB}/work-*-{tag}/nusel_evt*/nusel-evt*.tsv"):
        with open(t) as f:
            lines = [l.split() for l in f.read().splitlines() if l.strip()]
        body = lines[1:]
        lab = "evt" + os.path.basename(t)[len("nusel-evt"):-len(".tsv")]
        out[lab] = (len(body), f"{float(body[0][7]):.3f}" if body else None)
    return out


def main():
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(URL, timeout=60000)
        page.wait_for_selector(".slick-row", timeout=60000)
        page.wait_for_timeout(2000)

        # ---- case 1: in-beam-only, 1 -> 0 -> 1 --------------------------
        print("\n===== CASE 1: in-beam-only 287517 -> 287759 -> 287825")
        goto_event(page, "evt287517")
        page.click("button:has-text('Mode: ALL bundles')")
        page.wait_for_timeout(1500)
        a = show("287517 in-beam-only", page)
        next_event(page, "evt287759")
        b1 = show("287759 in-beam-only (expect NO rows)", page)
        next_event(page, "evt287825")
        c = show("287825 in-beam-only", page)
        if b1 and b1 == a:
            fails.append("C1: 287759 still shows 287517's row(s)")
        elif b1:
            fails.append(f"C1: 287759 shows {len(b1)} row(s), expected 0")
        if c == a:
            fails.append("C1: 287825 still shows 287517's row(s)")
        if not c:
            fails.append("C1: 287825 shows no rows, expected 1")

        # back to ALL mode
        page.click("button:has-text('Mode: IN-BEAM only')")
        page.wait_for_timeout(1500)

        # ---- case 2: ALL mode, 9 -> 9 -----------------------------------
        print("\n===== CASE 2: all bundles 286065 -> 286197 (9 -> 9)")
        goto_event(page, "evt286065")
        a2 = show("286065 all", page)
        next_event(page, "evt286197")
        b2 = show("286197 all", page)
        if a2 == b2:
            fails.append("C2: 286197 table identical to 286065 (stale cells)")
        if len(a2) != 9 or len(b2) != 9:
            fails.append(f"C2: row counts {len(a2)}/{len(b2)}, expected 9/9")

        # ---- case 3: label click, n -> n --------------------------------
        print("\n===== CASE 3: scan-label click repaints the row")
        # focus row 0, then toggle a label button
        page.click(".slick-row >> nth=0")
        page.wait_for_timeout(1200)
        before = show("286197 before label click", page)
        page.click("button:text-is('TGM')")
        page.wait_for_timeout(2000)
        after = show("286197 after label click", page)
        # The TGM button toggles, and the scratch tag may already carry the
        # label from an earlier run of this test, so assert the flip -- not a
        # fixed value.
        was, now = before[0][13], after[0][13]
        if before == after:
            fails.append("C3: row unchanged after a scan-label click (stale cells)")
        elif ("TGM" in now) == ("TGM" in was):
            fails.append(f"C3: scan cell {was!r} -> {now!r}, TGM did not toggle")

        # ---- case 4: walk all 30 events, compare against the TSVs --------
        print("\n===== CASE 4: walk every event in ALL mode vs the TSV")
        exp = expected_rows()
        labels = page.eval_on_selector("select >> nth=0",
                                       "s => [...s.options].map(o => o.value)")
        goto_event(page, labels[0])
        for k, lab in enumerate(labels):
            if k:
                next_event(page, lab)
            rs = rows(page)
            want_n, want_t = exp[lab]
            got_t = rs[0][2] if rs else None
            # SlickGrid renders only the visible viewport (height=290 => ~9
            # rows), so a 14-row event legitimately shows fewer DOM rows; the
            # identity that matters is row 0's flash time.
            ok = (got_t == want_t and (len(rs) == want_n or 8 <= len(rs) < want_n))
            print(f"    {lab}: rows={len(rs)}/{want_n} first_t={got_t}/{want_t} "
                  + ("ok" if ok else "MISMATCH"))
            if not ok:
                fails.append(f"C4 {lab}: rows={len(rs)} want {want_n}, "
                             f"first t={got_t} want {want_t}")
        b.close()

    print("\n===== RESULT")
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("PASS all four cases")
    return 0


sys.exit(main())

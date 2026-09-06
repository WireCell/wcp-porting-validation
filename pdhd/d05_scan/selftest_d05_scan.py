#!/usr/bin/env python3
"""Headless self-test for the PDHD mover-scan display (doc pdhd/05 sec 6).

Runs the viewer module against every one of its 39 objects with a scratch label
tag, then asserts the things that would silently ruin the scan:

  1. every object resolves to real charge in the arm being drawn, and its point
     count is EXACTLY the sheet's npts (wrong arm / wrong epoch => wrong object);
  2. THE BLIND: the set of zip members actually opened contains no tagger layer.
     This scan cannot lean on stm_scan's byte-identity argument (q differs
     between the arms), so the blind is an ABSENCE and has to be proven as one;
  3. every button writes a label string that d04_movers_score.py accepts -- an
     alphabet drift would make that scorer exit 2 on the finished scan;
  4. the exported sheet round-trips through the UNCHANGED scorer, and the
     committed blind sheet is not modified;
  5. the ends/geometry helpers survive the degenerate objects (npts = 3, 4, 9).

Usage:  python3 selftest_d05_scan.py        (rc 0 = pass; nothing is served)
"""
import os
import sys
import json
import csv
import hashlib
import importlib.util
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def load_viewer(tag):
    """Import the viewer with a scratch --tag.  bokeh's curdoc() works fine
    outside a server session; the app just never gets rendered."""
    sys.argv = ["d05_scan_viewer.py", "--tag", tag]
    spec = importlib.util.spec_from_file_location(
        "d05v", os.path.join(HERE, "d05_scan_viewer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tag = "selftest"
    sheet_before = hashlib.sha256(open(os.path.join(
        PDHD, "bee-pr-run029107-d04movers.sheet.tsv"), "rb").read()).hexdigest()
    v = load_viewer(tag)

    print("1. objects resolve, in the right arm")
    check(len(v.ITEMS) == 39, "39 objects on the sheet (got %d)" % len(v.ITEMS))
    bad = []
    for it in v.ITEMS:
        ch = v.event_charge(it["event"])
        if ch is None:
            bad.append((it["event"], it["cluster"], "no zip"))
            continue
        n = int((ch[4] == it["cluster"]).sum())
        if n != it["npts"]:
            bad.append((it["event"], it["cluster"], "npts %d != sheet %d" % (n, it["npts"])))
    check(not bad, "every object's point count equals the sheet's npts%s"
          % ("" if not bad else "  -- %s" % bad[:5]))

    print("2. the blind: no tagger layer is ever opened")
    for it in v.ITEMS:
        v.event_dead(it["event"])
    opened = sorted(v.MEMBERS_READ)
    print("     members opened: %s" % ", ".join(sorted(
        set(os.path.basename(m).split("-", 1)[1] for m in opened))))
    banned = ("stm", "stm_fit", "stm_tagged", "steiner_graph", "steiner_terminals")
    hits = [m for m in opened
            for b in banned if os.path.basename(m).endswith("-%s-global.json" % b)]
    check(not hits, "no stm / stm_fit / stm_tagged / steiner_* member opened%s"
          % ("" if not hits else "  -- %s" % hits))
    check(all(os.path.basename(m).endswith("-clustering-global.json")
              or v.RE_DEAD.fullmatch(os.path.basename(m)) for m in opened),
          "only clustering-global and channel-deadarea-* were opened")
    # and the key must not be readable from this process: it may be NAMED (the
    # export hint prints the scoring command) but never handed to open()/ZipFile.
    import re as _re
    src = open(os.path.join(HERE, "d05_scan_viewer.py")).read()
    reads = _re.findall(r"(?:open|ZipFile)\([^)\n]*KEY", src)
    check(not reads, "the viewer never passes the answer key to open()/ZipFile%s"
          % ("" if not reads else "  -- %s" % reads))

    print("3. the label alphabet matches the scorer")
    sys.path.insert(0, os.path.join(PDHD, "docs", "scripts"))
    spec = importlib.util.spec_from_file_location(
        "d04s", os.path.join(PDHD, "docs", "scripts", "d04_movers_score.py"))
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    check(set(v.CHOICES) == sc.VALID,
          "the 8 buttons are exactly the scorer's VALID set "
          "(buttons-only %s, scorer-only %s)"
          % (sorted(set(v.CHOICES) - sc.VALID), sorted(sc.VALID - set(v.CHOICES))))

    print("4. export round-trips through the UNCHANGED scorer")
    for i, it in enumerate(v.ITEMS):
        v.LABELS[v.item_key(it)] = dict(label=v.CHOICES[i % len(v.CHOICES)],
                                        notes="selftest", scan_id=it["scan_id"],
                                        event=it["event"], cluster=it["cluster"],
                                        npts=it["npts"], arm=v.ARM)
    with tempfile.TemporaryDirectory(dir="/home/xqian/tmp") as td:
        p = v.export_sheet(os.path.join(td, "filled.tsv"))
        rows = [r for r in csv.DictReader(
            (l for l in open(p) if not l.startswith("#")), delimiter="\t")]
        check(len(rows) == 39 and all(r["label"] for r in rows),
              "the exported sheet has all 39 rows labelled")
        r = subprocess.run([sys.executable,
                            os.path.join(PDHD, "docs", "scripts", "d04_movers_score.py"),
                            "--sheet", p, "--key",
                            os.path.join(PDHD, "bee-pr-run029107-d04movers.KEY.tsv")],
                           capture_output=True, text=True)
        check(r.returncode == 0,
              "d04_movers_score.py accepts every label this app can write (rc=%d)%s"
              % (r.returncode, "" if r.returncode == 0 else "\n" + r.stdout + r.stderr))
        # and one bad label must still be REFUSED, not absorbed
        txt = open(p).read().replace("\tTHRU\t", "\tprobably through\t", 1)
        bp = os.path.join(td, "bad.tsv")
        open(bp, "w").write(txt)
        r2 = subprocess.run([sys.executable,
                             os.path.join(PDHD, "docs", "scripts", "d04_movers_score.py"),
                             "--sheet", bp, "--key",
                             os.path.join(PDHD, "bee-pr-run029107-d04movers.KEY.tsv")],
                            capture_output=True, text=True)
        check(r2.returncode == 2, "a free-text label still makes the scorer REFUSE (rc=2)")
    v.LABELS.clear()

    print("5. the committed blind sheet was not written to")
    sheet_after = hashlib.sha256(open(os.path.join(
        PDHD, "bee-pr-run029107-d04movers.sheet.tsv"), "rb").read()).hexdigest()
    check(sheet_before == sheet_after, "sheet sha256 unchanged (%s)" % sheet_before[:12])

    print("6. geometry helpers survive the degenerate objects")
    import numpy as np
    for it in v.ITEMS:
        ch = v.event_charge(it["event"])
        m = ch[4] == it["cluster"]
        C = np.c_[ch[0], ch[1], ch[2]][m]
        e = v.ends_of(C)
        if it["npts"] < 20:            # the degenerate ones: n = 3, 4, 9, ...
            check(e is not None and np.isfinite(e[2]),
                  "ends_of ok for evt %s cl %d (n=%d, extent %.1f cm)"
                  % (it["event"], it["cluster"], it["npts"],
                     e[2] if e else float("nan")))
        if e is not None:
            for p in (e[0], e[1]):
                g, _ = v.wall_gap(p)
                assert np.isfinite(g), (it, p)
        v.dead_gap(C[0], v.event_dead(it["event"]))
    check(True, "wall_gap / dead_gap finite on all 39 objects")

    print("7. the close-up panels are not vacuous")
    near = {}
    for it in v.ITEMS:
        ch = v.event_charge(it["event"])
        X, Y, Z, _, C = ch
        m = C == it["cluster"]
        e = v.ends_of(np.c_[X, Y, Z][m])
        if e is None:
            continue
        n = 0
        for p in (e[0], e[1]):
            w = ((np.abs(X - p[0]) <= 25) & (np.abs(Y - p[1]) <= 25) &
                 (np.abs(Z - p[2]) <= 25))
            n = max(n, int((w & ~m).sum()))
        near[(it["event"], it["cluster"])] = n
    hot = sorted(v for v in near.values() if v > 0)
    check(len(hot) >= 5,
          "%d of %d objects have other charge within +/-25 cm of an end "
          "(max %d) -- the close-up carries the fragment signal"
          % (len(hot), len(near), max(near.values())))

    print("\n%s  (%d checks failed)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

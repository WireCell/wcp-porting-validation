#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 -- re-point the two pi0 denominator manifests at a pr150 arm.

FORK BY DUPLICATION (CLAUDE.md M10) of scripts/pr144_pi0_manifests.sh, which is
doc pr/144's record and stays byte-for-byte untouched.  What this fork changes:

  * arm naming.  pr144's awk rebuilds the dump path as
    "work-<sample>-<tag>/<evtdir>/<file>"; here the tag is `pr150<cell>` (or any
    --tag, so the reference arm d102mpr can be re-pointed the same way).
  * ABSOLUTE dump paths.  pr126_pi0_select.load_manifest() aborts the whole
    census when a manifest's dump ARM directory is missing -- a guard written
    for retired arms (doc pr/135 sec 11.2).  On a Stage-1 pr150 cell the mcp2k
    (and often mcp1k) arm genuinely does not exist yet, and a relative path
    would trip that guard and kill the run.  An absolute path has "" as its
    first path component, which the guard filters out, while
    SEL.load_json() passes an absolute path through unchanged and returns None
    for a missing file -- exactly the "absent on this arm" semantics the census
    already implements.  So the rows STAY IN the manifest and are MARKED.
  * an `absent` column (1 = no calib dump on this arm).  csv.DictReader keeps
    it; the census reads only `dump` and `sample`, so it is inert there.
  * it prints the reachable HAND-PI0 count per sample -- the number that
    actually sets the census denominator.  On a Stage-1 cell (nuecc48+ncpi0
    only) that is 14 of the 66, not 66.

READ-ONLY on the repo: writes only under --out (default
/home/xqian/tmp/pr150/scorers/manifests/<tag>/).

Usage:
    ./pi0_manifests.py --cell s0                 # -> work-<sample>-pr150s0
    ./pi0_manifests.py --tag d102mpr             # the committed reference arm
    ./pi0_manifests.py --cell csp3bw --out /home/xqian/tmp/pr150/scorers/manifests
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../sbnd_xin

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

# (set name, source manifest under em_display/, output basename).  These two
# committed manifests ARE the denominator -- which events, which hand labels.
# Only the `dump` column moves.
SETS = [("98",  "em117-132denom98-manifest.tsv",  "denom98.tsv"),
        ("141", "em114c-132denom141-manifest.tsv", "denom141.tsv")]

# The label sets doc pr/135's Repro block names.  Base labels first (they win),
# then the pr/132 pairing overlay that extends the denominator.
BASE_TAGS = ["emscan-0827", "emscan-0828-agent5"]
OVERLAY_TAG = "pi0scan-0829-agent"


def has_hand_pi0(rec):
    g = ((rec or {}).get("pio") or {}).get("gammas")
    return bool(g and all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", help="pr150 cell, e.g. s0 / cs / p3bw / csp3bw / tfull")
    ap.add_argument("--tag", help="full arm tag instead of a cell, e.g. d102mpr")
    ap.add_argument("--out", default="/home/xqian/tmp/pr150/scorers/manifests",
                    help="parent dir; the manifests land in <out>/<tag>/")
    a = ap.parse_args()
    if bool(a.cell) == bool(a.tag):
        ap.error("give exactly one of --cell or --tag")
    tag = a.tag or ("pr150" + a.cell)

    outdir = os.path.join(a.out, tag)
    os.makedirs(outdir, exist_ok=True)

    # hand-pi0 events, resolved the way pr132_pi0_census.py resolves them
    labels = {}
    for t in BASE_TAGS:
        labels[t] = SEL.load_labels(t)
    overlay = SEL.load_labels(OVERLAY_TAG)

    grand = Counter()
    for setname, src, dst in SETS:
        base = labels[BASE_TAGS[0] if setname == "98" else BASE_TAGS[1]]
        rows = list(csv.DictReader(open(os.path.join(SX, "em_display", src)),
                                   delimiter="\t"))
        out_rows = []
        per = {}
        for r in rows:
            ev, smp = int(r["event"]), r["sample"]
            arm = "work-%s-%s" % (smp, tag)
            dump = os.path.join(SX, arm, "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
            absent = 0 if os.path.exists(dump) else 1
            out_rows.append(dict(r, dump=dump, absent=absent))
            hand = has_hand_pi0(base.get(ev)) or has_hand_pi0(overlay.get(ev))
            c = per.setdefault(smp, Counter())
            c["rows"] += 1
            c["present"] += 1 - absent
            c["hand"] += 1 if hand else 0
            c["hand_present"] += 1 if (hand and not absent) else 0
            grand["rows"] += 1
            grand["present"] += 1 - absent
            grand["hand"] += 1 if hand else 0
            grand["hand_present"] += 1 if (hand and not absent) else 0

        path = os.path.join(outdir, dst)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, delimiter="\t",
                               fieldnames=["sample", "run", "subrun", "event",
                                           "dump", "absent"])
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print("%s-set -> %s" % (setname, path))
        for smp in sorted(per):
            c = per[smp]
            armdir = os.path.join(SX, "work-%s-%s" % (smp, tag))
            note = "" if os.path.isdir(armdir) else "   (arm dir does not exist)"
            print("   %-8s rows %3d  dump present %3d  absent %3d | hand pi0 %2d, "
                  "of those reachable %2d%s"
                  % (smp, c["rows"], c["present"], c["rows"] - c["present"],
                     c["hand"], c["hand_present"], note))

    print("TOTAL rows %d  present %d  absent %d | hand pi0 %d, reachable on %s: %d"
          % (grand["rows"], grand["present"], grand["rows"] - grand["present"],
             grand["hand"], tag, grand["hand_present"]))
    print("   (the census denominator is the REACHABLE count: an event with no "
          "calib dump on this arm is skipped by pr132_pi0_census.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

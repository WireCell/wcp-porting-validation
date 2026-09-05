#!/usr/bin/env python3
"""Score the STM hand-scan labels against both arms -- doc sec 13.

THIS reads the answer key (docs/scan/pdhd_retile_scan_key.tsv).  The viewer
never does.  Run it only after labelling, or the direction-of-change column
would be visible while you scan.

The procedure is fixed here BEFORE any label exists, for the same reason the
acceptance bar was stated before the scan: so it cannot be fitted to the result.

  For every labelled cluster with label in {STM, THRU} (UNCLEAR is counted and
  reported but scores neither arm), the human label is the truth and each arm's
  binary STM verdict is scored against it.  Reported per stratum
  (A: npts >= 200, B: npts < 200) and per direction of change, because the
  churn is not size-symmetric: 36% of the tags the knob ADDS sit in stratum B.

Usage:  python3 score_stm_scan.py [--tag retile0]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PDHD = os.path.dirname(HERE)
KEY = os.path.join(PDHD, "docs", "scan", "pdhd_retile_scan_key.tsv")


def load_key():
    with open(KEY) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    out = {}
    for r in csv.DictReader(lines, delimiter="\t"):
        out["%s/%s" % (r["event"], r["cluster"])] = dict(
            scan_id=int(r["scan_id"]), stratum=r["stratum"], npts=int(r["npts"]),
            off=int(r["stm_retiler_off"]), on=int(r["stm_retiler_on"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="retile0")
    a = ap.parse_args()
    lf = os.path.join(PDHD, "work", "stm_scan_labels", a.tag, "labels.json")
    if not os.path.isfile(lf):
        sys.exit("no labels at %s" % lf)
    labels = json.load(open(lf))["labels"]
    key = load_key()

    rows, unclear, missing = [], 0, 0
    for k, rec in labels.items():
        if k not in key:
            missing += 1
            continue
        if rec["label"] == "UNCLEAR":
            unclear += 1
            rows.append((key[k], rec, None))
            continue
        rows.append((key[k], rec, 1 if rec["label"] == "STM" else 0))

    scored = [(kk, rr, t) for kk, rr, t in rows if t is not None]
    print("labels: %d   scored: %d   UNCLEAR: %d   not in key: %d\n"
          % (len(labels), len(scored), unclear, missing))

    def block(name, sel):
        s = [x for x in scored if sel(x[0])]
        if not s:
            print("  %-34s (none)" % name)
            return
        off = sum(1 for kk, rr, t in s if kk["off"] == t)
        on = sum(1 for kk, rr, t in s if kk["on"] == t)
        u = sum(1 for kk, rr, t in rows if sel(kk) and t is None)
        print("  %-34s n=%3d   OFF right %3d (%.0f%%)   ON right %3d (%.0f%%)   "
              "net %+d   [UNCLEAR %d]"
              % (name, len(s), off, 100.0 * off / len(s), on, 100.0 * on / len(s),
                 on - off, u))

    print("BY STRATUM")
    block("A  npts >= 200", lambda k: k["stratum"] == "A")
    block("B  npts <  200", lambda k: k["stratum"] == "B")
    print("\nBY DIRECTION OF CHANGE")
    block("knob GAINS the tag (off0 -> on1)", lambda k: k["on"] == 1)
    block("knob LOSES the tag (off1 -> on0)", lambda k: k["on"] == 0)
    print("\nBY BOTH")
    for st in ("A", "B"):
        for d, lab in ((1, "gains"), (0, "loses")):
            block("%s, knob %s" % (st, lab),
                  lambda k, st=st, d=d: k["stratum"] == st and k["on"] == d)
    print("\nOVERALL")
    block("all scored", lambda k: True)

    print("\nUNCLEAR rate by stratum (a knob that tags fragments shows up here):")
    for st in ("A", "B"):
        tot = [x for x in rows if x[0]["stratum"] == st]
        un = [x for x in tot if x[2] is None]
        if tot:
            print("  %s: %d / %d (%.0f%%)" % (st, len(un), len(tot), 100.0 * len(un) / len(tot)))

    print("""
ACCEPTANCE BAR (stated in docs/scan/README.md before any label existed):
  flip only if the knob is net-positive in stratum A AND its stratum-B gains
  are not predominantly THRU/UNCLEAR.  A knob that fixes real tracks while
  inventing fragment tags is a different decision from one that does only the
  first -- read the 'B, knob gains' line and the stratum-B UNCLEAR rate.""")


if __name__ == "__main__":
    main()

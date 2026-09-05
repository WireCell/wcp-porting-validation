#!/usr/bin/env python3
"""Score the STM hand-scan labels against both arms -- doc sec 13.

THIS reads the answer key (docs/scan/pdhd_retile_scan_key.tsv).  The viewer
never does.  Run it only after labelling, or the direction-of-change column
would be visible while you scan.

The procedure is fixed BEFORE the labels exist, for the same reason the
acceptance bar was: so it cannot be fitted to the result.

  The human label is the truth and each arm's binary STM verdict is scored
  against it.  The label alphabet (sec 13.2) is

    STM / THRU              the cluster is the whole object; it stops / exits
    FRAG_STM / FRAG_THRU    the cluster is only PART of the object; the FULL
                            object stops / exits
    MESSY                   not one track -- "does it stop" is ill-posed
    UNCLEAR                 the scanner could not tell

  A FRAG row carries the same binary verdict as its plain counterpart, so
  under-clustering costs the scan no statistical power; partial=True is
  tallied separately as the under-clustering rate.  MESSY and UNCLEAR score
  neither arm and are reported as rates.

  Reported per stratum (A: npts >= 200, B: npts < 200) and per direction of
  change, because the churn is not size-symmetric: 36% of the tags the knob
  ADDS sit in stratum B.

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

# truth value contributed to the binary, or None for "scores neither arm".
# Any label absent from this table is a hard error -- a silent fall-through
# would have scored FRAG and MESSY as THRU.
TRUTH = {"STM": 1, "THRU": 0, "MESSY": None, "UNCLEAR": None}


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

    rows, missing = [], 0
    for k, rec in labels.items():
        lab = rec["label"]
        if lab not in TRUTH:
            sys.exit("unrecognised label %r on item %s -- refusing to score.\n"
                     "Add it to TRUTH with an explicit truth value first." % (lab, k))
        if k not in key:
            missing += 1
            continue
        rows.append((key[k], rec, TRUTH[lab], bool(rec.get("partial", False))))

    scored = [x for x in rows if x[2] is not None]
    nun = sum(1 for x in rows if x[1]["label"] == "UNCLEAR")
    nme = sum(1 for x in rows if x[1]["label"] == "MESSY")
    npart = sum(1 for x in rows if x[3])
    print("labels: %d   scored: %d   MESSY: %d   UNCLEAR: %d   not in key: %d"
          % (len(labels), len(scored), nme, nun, missing))
    print("of the scored rows, %d (%.0f%%) are FRAG -- the cluster was only part "
          "of the object\n" % (npart, 100.0 * npart / len(scored) if scored else 0))

    def block(name, sel):
        s = [x for x in scored if sel(x[0], x[3])]
        if not s:
            print("  %-34s (none)" % name)
            return
        off = sum(1 for kk, rr, t, p in s if kk["off"] == t)
        on = sum(1 for kk, rr, t, p in s if kk["on"] == t)
        u = sum(1 for kk, rr, t, p in rows if sel(kk, p) and t is None)
        fr = sum(1 for kk, rr, t, p in s if p)
        print("  %-34s n=%3d   OFF right %3d (%.0f%%)   ON right %3d (%.0f%%)   "
              "net %+d   [unscored %d, FRAG %d]"
              % (name, len(s), off, 100.0 * off / len(s), on, 100.0 * on / len(s),
                 on - off, u, fr))

    print("BY STRATUM")
    block("A  npts >= 200", lambda k, p: k["stratum"] == "A")
    block("B  npts <  200", lambda k, p: k["stratum"] == "B")
    print("\nBY DIRECTION OF CHANGE")
    block("knob GAINS the tag (off0 -> on1)", lambda k, p: k["on"] == 1)
    block("knob LOSES the tag (off1 -> on0)", lambda k, p: k["on"] == 0)
    print("\nBY BOTH")
    for st in ("A", "B"):
        for d, lab in ((1, "gains"), (0, "loses")):
            block("%s, knob %s" % (st, lab),
                  lambda k, p, st=st, d=d: k["stratum"] == st and k["on"] == d)
    print("\nOVERALL")
    block("all scored", lambda k, p: True)

    print("\nSENSITIVITY -- whole objects only (FRAG rows dropped)")
    block("all scored, partial excluded", lambda k, p: not p)
    block("A, gains, partial excluded",
          lambda k, p: k["stratum"] == "A" and k["on"] == 1 and not p)
    block("B, gains, partial excluded",
          lambda k, p: k["stratum"] == "B" and k["on"] == 1 and not p)

    print("\nWHAT THE KNOB IS ACTUALLY DOING -- composition of the tags it ADDS")
    g = [x for x in rows if x[0]["on"] == 1]
    if g:
        import collections
        c = collections.Counter((x[1].get("choice") or x[1]["label"]) for x in g)
        for name, n in sorted(c.items(), key=lambda kv: -kv[1]):
            print("  %-12s %3d  (%.0f%%)" % (name, n, 100.0 * n / len(g)))
        nf = sum(1 for x in g if x[3])
        print("  --> %d of %d gains (%.0f%%) are on an under-clustered fragment"
              % (nf, len(g), 100.0 * nf / len(g)))

    print("\nunscored rate by stratum (MESSY + UNCLEAR):")
    for st in ("A", "B"):
        tot = [x for x in rows if x[0]["stratum"] == st]
        un = [x for x in tot if x[2] is None]
        if tot:
            print("  %s: %d / %d (%.0f%%)" % (st, len(un), len(tot),
                                              100.0 * len(un) / len(tot)))

    print("""
ACCEPTANCE BAR
  ORIGINAL, stated in docs/scan/README.md before any label existed, quoted
  verbatim and NOT revised:

    "flip only if the knob is net-positive in stratum A AND its stratum-B gains
     are not predominantly THRU/UNCLEAR.  A knob that fixes real tracks while
     inventing fragment tags is a different decision from one that does only the
     first."

  APPENDED 2026-09-05, with exactly ONE label in existence (item 1, evt 21
  cl 125, UNCLEAR), when the scanner reported that some clusters are
  under-clustered pieces of a TGM and the FRAG/MESSY categories were added:

    The binary above is unchanged -- FRAG rows carry the full object's verdict,
    so they count in it.  Read 'BY BOTH' for the bar itself.  Additionally, and
    stated here before the labels exist: IF the FRAG share of the knob's GAINS
    is large, then the scan's finding is about under-clustering rather than
    about this knob, and that is the more important result even if the binary
    comes out favourable.  Report it either way; the flip is the owner's call.""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 -- re-score the pr/148 PID hand scan against a NEW arm.

FORK BY DUPLICATION (CLAUDE.md M10) of scripts/pr148_score_scan.py and
scripts/pr148_scan_combined.py (both are doc pr/148's records and stay
byte-for-byte).  What this fork changes, and why it had to:

  * pr148_score_scan.py builds `verdict_of` on `(event, shower_id)` with NO
    sample, and then indexes `by[k]` unconditionally.  On the arm it was made
    from (work-*-d145np, since RELEASED) every labelled object had a row; on a
    new arm an object can be absent, and the bare index raises KeyError -- the
    script cannot be pointed at a new arm as it stands.
  * it scores all THREE committed scan sets (pidscan / pidscan2 / pidscan3),
    keyed on (sample, event, shower_id), and reports the sample-less collision
    count the original keying would have suffered.
  * it runs an IDENTITY PROBE before scoring anything.  `shower_id` is a
    compact per-event reconstruction index and `obj` is the start segment's
    pf_node_id; neither survives a clustering change.  The verdict TSVs carry
    `obj`, `total_len_cm` and `kine_charge_mev` as recorded AT SCAN TIME, so
    the join can be checked instead of assumed
    (memory: feedback_scan_labels_need_their_source_arm).

READ-ONLY.  Prints; writes nothing.

Usage:
  ./pid_agreement.py --census /home/xqian/tmp/pr150/scorers/a5census-s0.tsv --label s0
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SCANS = [("scan0", "pr148-pidscan.KEY.tsv",  "pr148-pidscan-verdicts.tsv"),
         ("scan2", "pr148-pidscan2.KEY.tsv", "pr148-pidscan2-verdicts.tsv"),
         ("scan3", "pr148-pidscan3.KEY.tsv", "pr148-pidscan3-verdicts.tsv")]
LEN_TOL = 0.02          # fractional; "the same physical object" bar


def rd(p):
    with open(p) as fh:
        return list(csv.DictReader((l for l in fh if not l.startswith("#")),
                                   delimiter="\t"))


def F(r, k, d=None):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, help="A5 census TSV from pr148_a5_census.py")
    ap.add_argument("--label", default="arm")
    a = ap.parse_args()

    cen = rd(a.census)
    by = {}
    dup_nosample = Counter()
    for r in cen:
        by[(r["sample"], r["event"], r["shower_id"])] = r
        dup_nosample[(r["event"], r["shower_id"])] += 1
    coll = sum(1 for k, n in dup_nosample.items() if n > 1)
    samples = sorted({r["sample"] for r in cen})
    events = {(r["sample"], r["event"]) for r in cen}
    print("=== census %s: %d rows, %d showers re-typed, samples %s, %d events ==="
          % (a.label, len(cen), sum(1 for r in cen if r["verdict"] == "1"),
             ",".join(samples), len(events)))
    print("  (event, shower_id) keys shared by >1 sample: %d  -- keying WITHOUT the"
          " sample, as pr148_score_scan.py does, would merge these" % coll)

    # scan3 is the RESCAN of the SAME 36 objects scan0+scan2 labelled (doc pr/148
    # sec 15), so the three sets must never be summed: pass 1 = scan0+scan2,
    # pass 2 = scan3.  Verified here rather than assumed.
    tab = {"pass1": defaultdict(Counter), "pass2": defaultdict(Counter)}
    for name, keyf, labf in SCANS:
        pas = "pass2" if name == "scan3" else "pass1"
        key = {r["idx"]: r for r in rd(os.path.join(SX, "docs", "pr", keyf))}
        lab = [r for r in rd(os.path.join(SX, "docs", "pr", labf)) if r.get("verdict")]
        n = len(lab)
        matched = confirmed = obj_ok = len_ok = absent_evt = absent_row = 0
        for r in lab:
            k = (r["sample"], r["event"], r["shower_id"])
            row = by.get(k)
            if row is None:
                if (r["sample"], r["event"]) in events:
                    absent_row += 1        # event censused, this shower_id is not there
                else:
                    absent_evt += 1        # event not on this arm at all
                continue
            matched += 1
            o = (row["start_seg"] == r["obj"])
            L0, L1 = F(r, "total_len_cm"), F(row, "total_len_cm")
            l = (L0 is not None and L1 is not None and L0 > 0
                 and abs(L1 - L0) / L0 <= LEN_TOL)
            obj_ok += 1 if o else 0
            len_ok += 1 if l else 0
            if o and l:
                confirmed += 1
                tab[pas][r["verdict"]][row["verdict"]] += 1
        print("\n--- %s: %d labelled objects (KEY rows %d) ---" % (name, n, len(key)))
        print("  event not on this arm            : %d" % absent_evt)
        print("  event present, shower_id absent  : %d" % absent_row)
        print("  matched on (sample,event,shower) : %d" % matched)
        print("     of those, start_seg == obj    : %d" % obj_ok)
        print("     of those, total_len within %.0f%% : %d" % (100 * LEN_TOL, len_ok))
        print("  IDENTITY-CONFIRMED (both)        : %d  <- the only rows scored" % confirmed)

    for pas, what in (("pass1", "pass 1 = scan0 + scan2, 36 distinct objects"),
                      ("pass2", "pass 2 = scan3, the SAME 36 objects rescanned")):
        t = tab[pas]
        print("\n=== agreement on the identity-confirmed objects, %s ===" % what)
        print("  %-10s %11s %8s %6s" % ("hand", "A5 re-typed", "A5 kept", "total"))
        tot = Counter()
        for v in ("HADRONIC", "MIXED", "EM"):
            c = t.get(v, Counter())
            print("  %-10s %11d %8d %6d" % (v, c["1"], c["0"], c["1"] + c["0"]))
            tot["1"] += c["1"]
            tot["0"] += c["0"]
        print("  %-10s %11d %8d %6d" % ("TOTAL", tot["1"], tot["0"], tot["1"] + tot["0"]))
        ret = tot["1"]
        good = t.get("HADRONIC", Counter())["1"]
        if ret:
            print("  A5 precision on labelled re-types (HADRONIC / re-typed) : %d/%d = %.3f"
                  % (good, ret, 1.0 * good / ret))
        else:
            print("  A5 precision: no identity-confirmed re-type on this arm -- NOT 0,"
                  " simply not measurable here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc pr/88 -- filter out events that carry no hand-scannable vertex.

The owner, after scanning instalment 1: "there are many events in there, there
are only dots.  In this case, they are impossible to hand scan the true
vertices, so I just skipped them.  For our later fine tuning etc, we should
filter out this kind of events."

This is owner rule 8 in `scan_prompt.md` ("just a lot of dots -- it does not
matter") turned into a cut, and it matters for training rather than for
scanning: an event with no readable vertex still HAS a recorded label if
somebody clicks one, and that label is noise that a net will happily fit.

WHY A CUT ON `longest` AND NOT ON TOTAL LENGTH.  Validated against the 43
events of instalment 1, where the owner labelled 30 and skipped 13, so the
ground truth is the owner's own behaviour rather than a guess:

    longest fitted segment, owner SKIPPED (n=13):
        0.3 0.3 0.3 0.3 0.4 0.5 0.6 0.8 0.9 1.0 1.3 2.2 4.4
    longest fitted segment, owner LABELLED (n=30):
        2.4 3.5 7.4 19.4 24.7 30.1 34.5 45.0 ... 423.6

    cut                       catches skipped   catches labelled
    longest < 2.4 cm              12 / 13            0 / 30
    longest < 5.0 cm              13 / 13            2 / 30
    pr/80 sec3 step1              13 / 13            3 / 30
      (cluster total < 8 cm OR longest <= 3 cm)

`DEFAULT_LONGEST_CM = 5.0` is the operating point: it removes every event the
owner declined, at the cost of two the owner did label -- and both of those
(evt102247 at 2.4 cm, evt177360 at 3.5 cm) are events the blind AI scanner
had ALSO abstained on, i.e. the marginal cost is two labels nobody was
confident about.

pr/80's own sec 3 step 1 criterion is stricter and costs a third event,
evt318549 (cluster total 7.4 cm), which the owner labelled as part of the
calibration draw and which the scanner called `certain`.  That is the
difference between a rule tuned for "should the scanner abstain" and one
tuned for "should this event carry a training label"; this module implements
the second and says so.

The cut is deliberately NOT applied inside `selfscan.prepare` -- an event with
no readable vertex is still a legitimate thing to show a scanner (the owner's
position is that any click is as good as any other there), and silently
dropping events would move every denominator in this round's census.  It is a
filter for the TRAINING POOL, applied at `build_dataset.py` time.

Usage:
  python3 vtx_rules/scannability.py --dumps /home/xqian/tmp/pr88/mcp2k-scannable.txt
  python3 vtx_rules/scannability.py --dumps <file> --emit-keep /tmp/keep.txt
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scankit                                                   # noqa: E402

DEFAULT_LONGEST_CM = 5.0


def geometry(dump_path):
    """(largest cluster's total fitted length, longest single segment), cm.

    Reads the SANITIZED dump, so this is computable from exactly what a blind
    scanner is shown -- the cut can never depend on the reconstruction's own
    vertex answer.
    """
    d = scankit.sanitize(json.load(open(dump_path)))
    per = collections.defaultdict(float)
    longest = 0.0
    for s in d.get("segments") or []:
        L = float(s.get("length") or 0.0)
        per[s.get("cluster_id")] += L
        longest = max(longest, L)
    return (max(per.values()) if per else 0.0), longest


def unscannable(dump_path, longest_cm=DEFAULT_LONGEST_CM):
    """True if the event is 'only dots' -- no fitted object long enough to
    carry a direction, so no vertex can be read from it by anyone."""
    _, longest = geometry(dump_path)
    return longest < longest_cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True,
                    help="file of calib dump paths, one per line")
    ap.add_argument("--longest", type=float, default=DEFAULT_LONGEST_CM)
    ap.add_argument("--emit-keep", help="write the surviving dump paths here")
    ap.add_argument("--emit-drop", help="write the filtered-out events here")
    a = ap.parse_args()

    paths = [ln.strip() for ln in open(a.dumps)
             if ln.strip() and not ln.startswith("#")]
    keep, drop = [], []
    for p in paths:
        cl, lo = geometry(p)
        (drop if lo < a.longest else keep).append((p, cl, lo))

    print("%d events: %d scannable, %d 'only dots' at longest < %.1f cm (%.1f%%)"
          % (len(paths), len(keep), len(drop), a.longest,
             100.0 * len(drop) / max(1, len(paths))))
    for cut in (2.0, 3.0, 4.0, 5.0, 8.0):
        n = sum(1 for p, cl, lo in keep + drop if lo < cut)
        print("    at < %.0f cm: %3d (%.1f%%)" % (cut, n, 100.0*n/len(paths)))

    if a.emit_keep:
        with open(a.emit_keep, "w") as fh:
            for p, cl, lo in keep:
                fh.write(p + "\n")
        print("wrote %s (%d)" % (a.emit_keep, len(keep)))
    if a.emit_drop:
        with open(a.emit_drop, "w") as fh:
            fh.write("# event\tcluster_total_cm\tlongest_segment_cm\n")
            for p, cl, lo in sorted(drop, key=lambda r: r[2]):
                ev = "evt" + os.path.basename(p).split("evt")[-1].split(".")[0]
                fh.write("%s\t%.2f\t%.2f\n" % (ev, cl, lo))
        print("wrote %s (%d)" % (a.emit_drop, len(drop)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

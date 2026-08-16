#!/usr/bin/env python3
"""doc pr/89 sec 6 -- draw the REVIEW-agree calibration pile.

Every pr/88 fill tier selected on `not agrees`, so the 251 REVIEW-agreeing
events (scanner picked what the reconstruction picked, at likely/unclear
confidence) are the one stratum with ZERO owner labels and unknown precision
-- and at 30% of the arm it is the stratum that blows the inverse-propensity
baseline out to [53.3%, 83.0%] (doc pr/89 sec 1.2).  This script draws the
~40-event owner scan that collapses that interval to about +-3 points.

Unlike build_review_pile.py (whose tier-4 fill EXCLUDES agreeing events by
design), the whole pile here is one stratum, so there is nothing to blind the
draw against: the owner will know every served event is one where the scanner
agreed with the reconstruction.  That anchoring caveat is stated in the doc
rather than hidden.  The draw is written out BEFORE the owner scans, same
audit rule as build_review_pile.py's calibration tier.

Usage:
  python3 vtx_rules/ragree_pile.py \
      --runs vtx_rules/runs/mcp2k-20260816 \
      --out  vtx_rules/runs/mcp2k-ragree-20260816/pile
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scannability                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="scan run record with waves/*/review.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--labels-root", default=None,
                    help="vertex_labels/ dir; any event that already has an "
                         "owner label under any tag is skipped (default: "
                         "../vertex_labels relative to this script)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    waves = sorted(glob.glob(os.path.join(a.runs, "waves", "wave*",
                                          "review.json")))
    if not waves:
        raise SystemExit("no waves/*/review.json under %s" % a.runs)
    for w in waves:
        for r in json.load(open(w))["rows"]:
            r["wave"] = os.path.basename(os.path.dirname(w))
            rows.append(r)

    ra = [r for r in rows if r["bucket"] == "REVIEW" and r["agrees"]]
    dots = {r["event"] for r in ra if scannability.unscannable(r["dump"])}
    pool = [r for r in ra if r["event"] not in dots]

    # Belt-and-braces: a REVIEW-agree event should never carry an owner
    # label (no fill tier served them), but verify instead of assuming.
    lroot = a.labels_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "vertex_labels")
    labelled = set()
    for f in glob.glob(os.path.join(lroot, "*", "labels-evt*.json")):
        labelled.add("evt" + os.path.basename(f)[len("labels-evt"):-len(".json")])
    pre = [r for r in pool if r["event"] in labelled]
    pool = [r for r in pool if r["event"] not in labelled]

    print("REVIEW-agree: %d total, %d 'only dots' dropped, %d already "
          "labelled (unexpected: %s), %d in the draw pool"
          % (len(ra), len(dots), len(pre),
             ", ".join(r["event"] for r in pre) or "none", len(pool)))

    rng = random.Random(a.seed)
    n = min(a.n, len(pool))
    draw = sorted(rng.sample(sorted(pool, key=lambda r: r["event"]), n),
                  key=lambda r: r["event"])

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pile-dumps.txt"), "w") as fh:
        for r in draw:
            fh.write(os.path.abspath(r["dump"]) + "\n")
    json.dump(dict(seed=a.seed, n=n, drawn_from_n=len(pool),
                   stratum="REVIEW-agree (bucket=REVIEW, agrees=true)",
                   dropped_only_dots=sorted(dots),
                   already_labelled=[r["event"] for r in pre],
                   events=[r["event"] for r in draw],
                   purpose="doc pr/89 sec 6: close the sec 1.2 IPW baseline "
                           "(REVIEW-agree stratum precision)"),
              open(os.path.join(a.out, "draw.json"), "w"), indent=1)
    json.dump(dict(served=n,
                   order=[dict(i=i, event=r["event"], tier="ragree",
                               bucket=r["bucket"], conf=r["conf"],
                               vertex_id=r["vertex_id"],
                               reco_sep_cm=r["reco_sep_cm"],
                               wave=r["wave"], why=r["why"], dump=r["dump"])
                          for i, r in enumerate(draw)]),
              open(os.path.join(a.out, "pile.json"), "w"), indent=1)
    print("wrote %s/{pile-dumps.txt,draw.json,pile.json}" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

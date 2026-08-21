#!/usr/bin/env python3
"""doc 74 -- cosmic_consistent_fv knob census: cosmict flags + BDT scores, arm A vs arm B.

Standalone (nearest kin: pr85_hash_gate.py's arm-walking; the comparison itself is
new because no prior round touched the cosmict_* block).  For every event present
in BOTH arms, reads pr_evt<ID>/calib-pr-evt<ID>.json and compares:
  - tagger.cosmict_flag and tagger.cosmict_flag_{1..9}, cosmict_flag_10_any
  - tagger.numu_score, tagger.nue_score (the BDT-input consequence)
Prints a per-event row for every difference and a per-flag OFF->ON count summary.

Usage: d74_cosmict_census.py <armOFF> <armON> [--label name]
Exit 0 always (census, not a gate).
"""
import argparse
import glob
import json
import os
import re
import sys

FLAGS = ["cosmict_flag"] + [f"cosmict_flag_{i}" for i in range(1, 10)] + ["cosmict_flag_10_any"]
SCORES = ["numu_score", "nue_score"]


def events_of(arm):
    out = {}
    for p in glob.glob(os.path.join(arm, "pr_evt*", "calib-pr-evt*.json")):
        m = re.search(r"calib-pr-evt(\d+)\.json$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def tagger_of(path):
    with open(path) as f:
        return json.load(f).get("tagger", {}) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA", help="OFF arm")
    ap.add_argument("armB", help="ON arm")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ea, eb = events_of(args.armA), events_of(args.armB)
    common = sorted(set(ea) & set(eb))
    print(f"# {args.label} events: A={len(ea)} B={len(eb)} common={len(common)}")

    cnt_a = {f: 0 for f in FLAGS}
    cnt_b = {f: 0 for f in FLAGS}
    n_diff_evt = 0
    n_score_evt = 0
    for evt in common:
        ta, tb = tagger_of(ea[evt]), tagger_of(eb[evt])
        diffs = []
        for f in FLAGS:
            va, vb = int(bool(ta.get(f, 0))), int(bool(tb.get(f, 0)))
            cnt_a[f] += va
            cnt_b[f] += vb
            if va != vb:
                diffs.append(f"{f}:{va}->{vb}")
        sc = []
        for s in SCORES:
            va, vb = float(ta.get(s, 0) or 0), float(tb.get(s, 0) or 0)
            if abs(va - vb) > 1e-9:
                sc.append(f"{s}:{va:.4f}->{vb:.4f}")
        if diffs or sc:
            print(f"evt {evt}: " + " ".join(diffs + sc))
            if diffs:
                n_diff_evt += 1
            if sc:
                n_score_evt += 1
    print(f"# summary {args.label}: flag-diff events {n_diff_evt}/{len(common)}, "
          f"score-diff events {n_score_evt}/{len(common)}")
    for f in FLAGS:
        if cnt_a[f] or cnt_b[f]:
            print(f"#   {f}: {cnt_a[f]} -> {cnt_b[f]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

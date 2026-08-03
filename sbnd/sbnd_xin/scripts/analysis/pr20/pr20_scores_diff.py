#!/usr/bin/env python3
"""Physics-column diff of two pr_scores_table.py TSVs (doc pr/20 gate PI-6).

A knob that only records or only produces a verdict must not move a single
physics number.  Hash gating cannot check that here -- P1 adds a `perblob`
array and P2 a cluster flag that `normalize_cluster_flags` materialises on
every cluster, so the archives legitimately differ -- so the bar is the score
table's physics columns instead.

Timing and memory columns are excluded BY NAME (not by a heuristic): wall_s,
core_s, timecmd_wall_s and maxrss_kb vary run to run on an unloaded box and
carry no physics.  Everything else is compared exactly as a string, so an
integer that becomes a float counts as a difference.

Exit status: 0 iff every compared cell matches on every event present in both.

Repro:
  ./pr20_scores_diff.py off.tsv on.tsv
  ./pr20_scores_diff.py off.tsv on.tsv --only numu_score kine_reco_Enu_MeV
"""
import argparse
import sys

EXCLUDE = {"sample", "wall_s", "core_s", "timecmd_wall_s", "maxrss_kb"}


def load(path):
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        rows = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) != len(hdr):
                continue
            d = dict(zip(hdr, f))
            rows[d.get("event")] = d
    return hdr, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--only", nargs="+", default=None,
                    help="compare only these columns")
    ap.add_argument("--max-print", type=int, default=40)
    args = ap.parse_args()

    hdr_a, A = load(args.a)
    hdr_b, B = load(args.b)
    if hdr_a != hdr_b:
        print("FAIL: headers differ", file=sys.stderr)
        return 2

    cols = args.only if args.only else [c for c in hdr_a if c not in EXCLUDE]
    common = sorted(set(A) & set(B), key=lambda e: int(e) if e and e.isdigit() else 0)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))

    diffs = []
    for evt in common:
        for c in cols:
            if A[evt].get(c) != B[evt].get(c):
                diffs.append((evt, c, A[evt].get(c), B[evt].get(c)))

    print("events: %d in both, %d only in A, %d only in B"
          % (len(common), len(only_a), len(only_b)))
    print("columns compared: %d  (excluded: %s)"
          % (len(cols), ", ".join(sorted(EXCLUDE & set(hdr_a)))))
    if only_a:
        print("  only in A: %s" % " ".join(only_a[:20]))
    if only_b:
        print("  only in B: %s" % " ".join(only_b[:20]))

    if not diffs:
        print("PASS: 0 differing cells over %d event(s) x %d column(s)"
              % (len(common), len(cols)))
        return 0 if not (only_a or only_b) else 1

    ev = sorted({d[0] for d in diffs}, key=lambda e: int(e))
    bycol = {}
    for _, c, _, _ in diffs:
        bycol[c] = bycol.get(c, 0) + 1
    print("FAIL: %d differing cell(s) on %d event(s)" % (len(diffs), len(ev)))
    print("  by column: %s" % ", ".join("%s=%d" % kv for kv in sorted(bycol.items())))
    for evt, c, va, vb in diffs[:args.max_print]:
        print("    evt %s %s: %r -> %r" % (evt, c, va, vb))
    if len(diffs) > args.max_print:
        print("    ... %d more" % (len(diffs) - args.max_print))
    return 1


if __name__ == "__main__":
    sys.exit(main())

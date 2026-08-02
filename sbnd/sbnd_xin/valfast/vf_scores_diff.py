#!/usr/bin/env python3
"""valfast gate 2: physics-column diff of two pr_scores_table.py TSVs.

Fork of pr20_scores_diff.py (kept untouched -- doc pr/20 gate PI-6 cites it)
with ONE deliberate difference: `kine_reco_Enu_MeV` is compared with a
RELATIVE TOLERANCE of 1e-5 instead of string equality. Reason: the PR chain
has a documented run-to-run noise floor in kine_reco_Enu ONLY -- last-digit
float flutter that survives `setarch x86_64 -R` (~7 cells / 1000 events; doc
pr/20 Part I census; re-measured 2026-08-02 on the valfast vfaa1/vfaa2 A/A'
arms: 7 cells / 629 events, max relative difference 2e-7). Every other
column is exact-string, as in the original. A real energy-scale change moves
Enu by orders of magnitude more than 1e-5 relative.

Exit status: 0 iff every compared cell matches (within the Enu tolerance) on
every event present in both.
"""
import argparse
import sys

EXCLUDE = {"sample", "wall_s", "core_s", "timecmd_wall_s", "maxrss_kb"}
RELTOL = {"kine_reco_Enu_MeV": 1e-5}


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


def cell_equal(col, va, vb):
    if va == vb:
        return True
    tol = RELTOL.get(col)
    if tol is None:
        return False
    try:
        fa, fb = float(va), float(vb)
    except (TypeError, ValueError):
        return False
    scale = max(abs(fa), abs(fb), 1e-30)
    return abs(fa - fb) / scale <= tol


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

    diffs, tolerated = [], 0
    for evt in common:
        for c in cols:
            va, vb = A[evt].get(c), B[evt].get(c)
            if va != vb and cell_equal(c, va, vb):
                tolerated += 1
            elif va != vb:
                diffs.append((evt, c, va, vb))

    print("events: %d in both, %d only in A, %d only in B"
          % (len(common), len(only_a), len(only_b)))
    print("columns compared: %d  (excluded: %s; rel-tol %s)"
          % (len(cols), ", ".join(sorted(EXCLUDE & set(hdr_a))),
             ", ".join("%s<=%g" % kv for kv in sorted(RELTOL.items()))))
    if tolerated:
        print("  %d cell(s) within the kine_reco_Enu noise floor (tolerated)" % tolerated)
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

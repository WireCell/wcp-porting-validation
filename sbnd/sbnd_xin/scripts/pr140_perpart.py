#!/usr/bin/env python3
"""doc pr/139 item 2 -- the MERGED (per-part) completeness target, summarised.

Reads the two per-part TSVs em140_score.py writes for an arm and reports the
same headline shape pr136_completeness.py reports for the single-target metric,
so the two can be quoted side by side.  Optionally diffs against a baseline arm.

    python3 scripts/pr140_perpart.py --arm on --base BASE
    python3 scripts/pr140_perpart.py --arm on --base BASE --tsv docs/pr/pr140-perpart-delta.tsv

The number that only THIS metric can produce is `parts with no reco match`:
a hand-labelled part that no reconstructed shower claims, i.e. a cut the owner
confirmed and the reco did not make.  Under the single-target metric that
failure is invisible -- the un-split object matches the one target and scores
well.  It is why doc pr/139 P1.4 (re-home) had no metric that could see it.
"""
import argparse, csv, os, statistics, sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(SX, "docs", "pr")


def load(tag):
    rows = []
    for s in ("98", "141"):
        p = os.path.join(D, "pr140-perpart-%s-%s.tsv" % (tag, s))
        if not os.path.exists(p):
            sys.exit("missing %s -- run scripts/pr140_score.sh %s first" % (p, tag))
        for r in csv.DictReader(open(p), delimiter="\t"):
            r["set"] = s
            rows.append(r)
    return rows


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return float("nan")


def summarise(tag, rows):
    qt = sum(num(r, "q_target") for r in rows)
    qm = sum(num(r, "q_miss") for r in rows)
    qx = sum(num(r, "q_extra") for r in rows)
    f1 = [num(r, "q_f1") for r in rows if num(r, "q_f1") == num(r, "q_f1")]
    parts = [r for r in rows if r["part"] not in ("-", "*")]
    resid = [r for r in rows if r["part"] == "*"]
    # A part whose intersection with the completeness target is empty is not a
    # failed cut -- the completeness scan had already excluded it (see the note
    # in em140_score.py).  Count it separately or the metric invents failures.
    empty = [r for r in parts if num(r, "q_target") == 0]
    live = [r for r in parts if num(r, "q_target") > 0]
    nomatch = [r for r in live if r["matched"] == "-1" or num(r, "q_comp") == 0]
    pf1 = [num(r, "q_f1") for r in live if num(r, "q_f1") == num(r, "q_f1")]
    out = dict(tag=tag, n_rows=len(rows), q_target=qt,
               q_miss=qm, q_miss_pct=100 * qm / qt if qt else float("nan"),
               q_extra=qx, q_extra_pct=100 * qx / qt if qt else float("nan"),
               med_f1=statistics.median(f1) if f1 else float("nan"),
               mean_f1=statistics.fmean(f1) if f1 else float("nan"),
               n_parts=len(live), n_empty=len(empty), n_resid=len(resid),
               n_nomatch=len(nomatch),
               med_part_f1=statistics.median(pf1) if pf1 else float("nan"),
               nomatch_rows=nomatch)
    return out


def show(s):
    print("  rows scored (showers + parts + residuals) : %d" % s["n_rows"])
    print("  sum q_target                              : %.4g" % s["q_target"])
    print("  q_miss  (UNDER)                           : %.4g = %.1f%%"
          % (s["q_miss"], s["q_miss_pct"]))
    print("  q_extra (OVER)                            : %.4g = %.1f%%"
          % (s["q_extra"], s["q_extra_pct"]))
    print("  charge-weighted F1 per row                : median %.3f  mean %.3f"
          % (s["med_f1"], s["mean_f1"]))
    print("  hand-labelled PARTS scored (non-empty)    : %d   median q_f1 %.3f"
          % (s["n_parts"], s["med_part_f1"]))
    print("  parts already excluded by the scan        : %d   (empty target -- not a failed cut)"
          % s["n_empty"])
    print("  residual '*' rows                         : %d" % s["n_resid"])
    print("  PARTS WITH NO RECO MATCH                  : %d   <- confirmed cuts the reco did not make"
          % s["n_nomatch"])
    for r in sorted(s["nomatch_rows"], key=lambda r: (int(r["event"]), int(r["shower"]), r["part"])):
        print("      evt%-8s shower %-7s part %-3s  q_target %.4g"
              % (r["event"], r["shower"], r["part"], num(r, "q_target")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base")
    ap.add_argument("--tsv")
    a = ap.parse_args()

    A = summarise(a.arm, load(a.arm))
    print("=== MERGED (per-part) completeness target -- arm %s ===" % a.arm)
    show(A)
    if a.base:
        B = summarise(a.base, load(a.base))
        print("\n=== baseline %s ===" % a.base)
        show(B)
        print("\n=== delta (%s - %s) ===" % (a.arm, a.base))
        print("  q_miss  %+.2f pt      q_extra %+.2f pt      med part q_f1 %+.3f"
              % (A["q_miss_pct"] - B["q_miss_pct"],
                 A["q_extra_pct"] - B["q_extra_pct"],
                 A["med_part_f1"] - B["med_part_f1"]))
        print("  parts with no reco match: %d -> %d  (%+d)"
              % (B["n_nomatch"], A["n_nomatch"], A["n_nomatch"] - B["n_nomatch"]))
        bk = {(r["event"], r["shower"], r["part"]) for r in B["nomatch_rows"]}
        ak = {(r["event"], r["shower"], r["part"]) for r in A["nomatch_rows"]}
        if bk - ak:
            print("    CUTS NOW MADE   : %s" % sorted(bk - ak))
        if ak - bk:
            print("    CUTS NOW MISSED : %s" % sorted(ak - bk))
        if a.tsv:
            with open(a.tsv, "w", newline="") as fh:
                w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                w.writerow(["metric", a.base, a.arm, "delta"])
                for k in ("q_miss_pct", "q_extra_pct", "med_part_f1", "n_nomatch", "n_parts"):
                    w.writerow([k, "%.4g" % B[k], "%.4g" % A[k], "%.4g" % (A[k] - B[k])])
            print("\nwrote %s" % a.tsv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

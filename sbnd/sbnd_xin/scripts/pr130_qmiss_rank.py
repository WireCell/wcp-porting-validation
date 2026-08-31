#!/usr/bin/env python3
"""doc pr/130 item 1 -- refreshed q_miss ranking at the TRUE production point.

Answers the go/no-go the pr/128 hand-off deferred ("item 3, q_miss hand-look"):
is charge error concentrated enough in a handful of events that an
under-clustering hand-scan round is worth running?

Two DIFFERENT >70% questions live here and they do not agree -- report both:

  (a) PREMISE     is q_miss the majority of total charge error?
                  ("75% of charge error is q_miss" -- pr/128 hand-off)
  (b) CONCENTRATION does the top-10 hold >70% of q_miss after adjudicated
                  rows are crossed off?  This is the one that decides whether
                  a top-10 hand-scan is worth a scanner's time.

Adjudicated rows are crossed off because their reconstruction is already
owner-ruled: scoring them again cannot produce a new action.  179369 and
283515 are excluded for a second reason -- both are `stem_backfill_back_guard`
movers (pr/130 Part 4), so their q_miss is measured against a shape the owner
has ruled on but production has not yet been changed to match.

Repro:
  scripts/pr130_qmiss_rank.py                       # both default tables
  scripts/pr130_qmiss_rank.py A.tsv B.tsv
"""
import csv
import sys
from collections import defaultdict

# event -> why it can no longer motivate a hand-scan
ADJUDICATED = {
    "318769": "pr/129 owner reject (141-rank-1)",
    "415278": "pr/124 declined trade-off",
    "179369": "pr/130 Part 4 -- owner OFF better (back-guard contested)",
    "283515": "pr/130 Part 4 -- owner ON better (back-guard mover)",
}

DEFAULTS = [
    ("98-set  (labels emscan-0827, em114 family)",
     "docs/pr/pr130-98-score-prod.tsv"),
    ("141-set (labels emscan-0828-agent5, em114c)",
     "docs/pr/pr130-141-score-prod.tsv"),
]


def per_event(path):
    """Sum the per-shower rows of an em117_score.py --tsv table per event."""
    miss, extra = defaultdict(float), defaultdict(float)
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            miss[r["event"]] += float(r["q_miss"])
            extra[r["event"]] += float(r["q_extra"])
    return miss, extra


def report(name, path, topn=10):
    miss, extra = per_event(path)
    tot_m, tot_x = sum(miss.values()), sum(extra.values())
    share_premise = 100.0 * tot_m / (tot_m + tot_x)

    print("=" * 78)
    print("%s\n  table: %s   (%d events carry marked rows)"
          % (name, path, len(miss)))
    print("  total q_miss  = %.4g" % tot_m)
    print("  total q_extra = %.4g" % tot_x)
    print("  (a) PREMISE  q_miss share of charge error = %.1f%%  %s"
          % (share_premise, "PASS >70%" if share_premise > 70 else "FAIL <70%"))

    rank = sorted(miss.items(), key=lambda kv: -kv[1])
    print("\n  rank  event      q_miss     cum%   adjudicated")
    for i, (ev, v) in enumerate(rank[:15], 1):
        cum = 100.0 * sum(x for _, x in rank[:i]) / tot_m
        print("  %4d  %-8s %9.4g  %6.1f   %s" % (i, ev, v, cum,
                                                 ADJUDICATED.get(ev, "")))

    kept = [(e, v) for e, v in rank if e not in ADJUDICATED]
    gone = [(e, v) for e, v in rank if e in ADJUDICATED]
    tot_k = sum(v for _, v in kept)
    print("\n  crossed off %d adjudicated event(s), %.4g q_miss (%.1f%% of set):"
          % (len(gone), sum(v for _, v in gone),
             100.0 * sum(v for _, v in gone) / tot_m))
    for ev, v in gone:
        print("     %-8s %9.4g   %s" % (ev, v, ADJUDICATED[ev]))

    kept_x = sum(v for e, v in extra.items() if e not in ADJUDICATED)
    share_premise_kept = (100.0 * tot_k / (tot_k + kept_x)
                          if (tot_k + kept_x) else 0.0)
    top = sum(v for _, v in kept[:topn])
    share_conc = 100.0 * top / tot_k if tot_k else 0.0
    print("\n  re-totalled q_miss (adjudicated removed) = %.4g" % tot_k)
    print("  (a) PREMISE re-checked on the kept pool  = %.1f%%  %s"
          % (share_premise_kept,
             "PASS >70%" if share_premise_kept > 70 else "FAIL <70%"))
    print("  (b) CONCENTRATION  top-%d share = %.1f%%  %s"
          % (topn, share_conc, "PASS >70%" if share_conc > 70 else "FAIL <70%"))
    print("      top-%d: %s" % (topn, ", ".join(e for e, _ in kept[:topn])))
    print("      NOTE: top-%d is %.0f%% of the %d KEPT events (denominator is"
          % (topn, 100.0 * topn / len(kept), len(kept)))
    print("            the post-crossoff pool, matching the (b) numerator).")
    return share_premise, share_conc


def main():
    args = sys.argv[1:]
    sets = ([("table %d" % i, p) for i, p in enumerate(args, 1)]
            if args else DEFAULTS)
    out = [report(name, path) for name, path in sets]
    print("=" * 78)
    prem = ["%.1f%%" % p for p, _ in out]
    conc = ["%.1f%%" % c for _, c in out]
    print("(a) PREMISE       q_miss > half the charge error: %s" % " / ".join(prem))
    print("(b) CONCENTRATION top-10 > 70%% of q_miss:        %s" % " / ".join(conc))
    print("GO/NO-GO on (b), the hand-scan criterion: %s"
          % ("GO -- passes on BOTH sets" if all(c > 70 for _, c in out)
             else "NO-GO -- fails on at least one set"))


if __name__ == "__main__":
    main()

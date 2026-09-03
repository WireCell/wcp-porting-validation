#!/usr/bin/env python3
"""doc 99 round 2 -- did T_cluster resolve the flash the cluster actually matched?

THE INSTRUMENT.  tracking-pr.root:T_cluster carries both halves of the answer on
the SAME row: cluster_t0_us, written from the flash QLMatching matched the
cluster to, and flash_time_us, written from whatever the reader resolved.  They
are the same double divided by the same constant, so

    flash_time_us == cluster_t0_us            (exact, not approximate)

is a within-file identity that needs no baseline arm, no archive census and no
cross-stage join.  That last point is why this exists: doc 99's first round
predicted moved rows from an archive census and had to learn the hard way that
the Q/L-stage census cannot see the PR stage's re-clustering
(feedback_diff_tool_cannot_prove_containment).  This tool reads one file and
asks it about itself.

"This cluster matched a flash" means t0 is neither 0 nor the NEVER-MATCHED
SENTINEL.  QLMatching stamps every cluster with the triple
(cluster_t0 = -1e12 ns, flash = -1, matched_flash_gid = -1) before matching
starts (QLMatching.cxx:1351), and a cluster nothing matches keeps it; -1e12 ns
is -1e9 in the us this tree stores.  Missing that cost a false FAIL on the first
run of this tool: 19 of 20175 rows were counted as "matched but unresolved" when
they are clusters that were never matched at all, and BOTH fixes were right to
resolve them to nothing.  The sentinel is counted and printed, never silently
dropped.

  CORRECT   matched and flash_id >= 0 and flash_time_us == cluster_t0_us
  WRONG     matched and flash_id >= 0 and flash_time_us != cluster_t0_us  <- the defect
  MISSING   matched but flash_id < 0            (the reader resolved nothing)
  ORPHAN    not matched but flash_id >= 0       (a flash for an unmatched cluster)
  unset     t0 == the sentinel                  (never matched; not in the denominator)
  zero      t0 == 0 exactly                     (not in the denominator)

Reported, never asserted: a cluster whose genuine t0 is exactly 0 would be
counted unmatched.  On SBND that is a flash at the trigger to the last bit.

Repro:
  python3 scripts/analysis/d99_tcluster_flash_check.py d99r2rdpr \\
      --samples ncpi0,nuecc48,mcp1k

  # two arms resolving by INDEPENDENT paths must agree row for row
  python3 scripts/analysis/d99_tcluster_flash_check.py d99r2wrpr \\
      --compare d99r2bothpr --samples ncpi0,nuecc48,mcp1k

Exit 0 iff every arm read at least one event AND (with --require-correct) every
matched row is CORRECT AND (with --compare) every joined row agrees.
"""
import argparse, os, re, sys

import numpy as np
import uproot

COLS = ["cluster_id", "flash_id", "flash_time_us", "flash_pe", "cluster_t0_us"]

# QLMatching.cxx:1351 pre-stamps every cluster with cluster_t0 = -1e12 (WCT ns)
# alongside flash = -1 / matched_flash_gid = -1; T_cluster divides by units::us,
# so an unmatched cluster reads exactly this.  Compared exactly, not with a
# tolerance: it is a literal, not a measurement.
UNSET_T0_US = -1e9


def events_of(arm_dir):
    return sorted(int(m.group(1)) for m in
                  (re.match(r"pr_evt(\d+)$", d) for d in os.listdir(arm_dir)) if m)


def read(path):
    with uproot.open(path) as f:
        if "T_cluster" not in f:
            return None
        t = f["T_cluster"]
        if not set(COLS) <= set(t.keys()):
            return None
        return t.arrays(COLS, library="np")


def scan(root, arm, samples):
    """-> (per-event rows, counters). Rows: (sample, event, cid, cls, t0, ft, pe)."""
    tally = dict(events=0, rows=0, matched=0, correct=0, wrong=0, missing=0,
                 orphan=0, unset=0, zero=0)
    detail = []
    for smp in samples:
        d = os.path.join(root, "work-%s-%s" % (smp, arm))
        if not os.path.isdir(d):
            print("MISSING ARM: %s" % d)
            tally["missing_arm"] = tally.get("missing_arm", 0) + 1
            continue
        for e in events_of(d):
            p = os.path.join(d, "pr_evt%d" % e, "tracking-pr.root")
            if not os.path.exists(p):
                continue
            a = read(p)
            if a is None:
                print("  %s evt%d: no usable T_cluster" % (smp, e))
                tally["missing_arm"] = tally.get("missing_arm", 0) + 1
                continue
            tally["events"] += 1
            fid, ft, t0 = a["flash_id"], a["flash_time_us"], a["cluster_t0_us"]
            n = len(fid)
            tally["rows"] += n
            for i in range(n):
                if t0[i] == UNSET_T0_US:
                    tally["unset"] += 1
                elif t0[i] == 0.0:
                    tally["zero"] += 1
                has_t0 = (t0[i] != 0.0 and t0[i] != UNSET_T0_US)
                has_f = (fid[i] >= 0)
                if has_t0:
                    tally["matched"] += 1
                if has_t0 and has_f:
                    cls = "CORRECT" if ft[i] == t0[i] else "WRONG"
                elif has_t0:
                    cls = "MISSING"
                elif has_f:
                    cls = "ORPHAN"
                else:
                    continue
                tally[cls.lower()] = tally.get(cls.lower(), 0) + 1
                detail.append((smp, e, int(a["cluster_id"][i]), cls,
                               float(t0[i]), float(ft[i]), float(a["flash_pe"][i])))
    return detail, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="arm suffix, e.g. d99r2rdpr")
    ap.add_argument("--compare", default=None,
                    help="second arm; join on (sample,event,cluster_id) and require "
                         "flash_time_us and flash_pe to agree exactly")
    ap.add_argument("--samples", default="ncpi0,nuecc48,mcp1k")
    ap.add_argument("--require-correct", action="store_true",
                    help="exit non-zero unless every matched row is CORRECT")
    ap.add_argument("--out", default=None, help="write the per-row detail TSV here")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    args = ap.parse_args()
    samples = [s for s in args.samples.split(",") if s]

    detail, t = scan(args.root, args.arm, samples)
    rc = 0
    if t["events"] == 0 or t.get("missing_arm"):
        # A gate that passes on no data is worse than no gate (doc 99 round 1).
        print("REFUSE: %d events read, %d missing/unreadable arms -- nothing was tested"
              % (t["events"], t.get("missing_arm", 0)))
        return 1

    den = t["matched"]
    pct = (100.0 * t["correct"] / den) if den else 0.0
    print("arm %-12s events=%-5d T_cluster rows=%-7d matched(t0!=0)=%-7d"
          % (args.arm, t["events"], t["rows"], den))
    print("  CORRECT %7d (%.1f%%)   WRONG %7d   MISSING %6d   ORPHAN %6d"
          % (t["correct"], pct, t["wrong"], t["missing"], t["orphan"]))
    print("  not in the denominator: %d never-matched (t0 sentinel), %d t0 == 0"
          % (t["unset"], t["zero"]))

    if args.out:
        with open(args.out, "w") as fp:
            fp.write("sample\tevent\tcluster_id\tclass\tcluster_t0_us\tflash_time_us\tflash_pe\n")
            for r in detail:
                fp.write("%s\t%d\t%d\t%s\t%.9g\t%.9g\t%.9g\n" % r)
        print("  detail -> %s (%d rows)" % (args.out, len(detail)))

    if args.require_correct and (t["wrong"] or t["missing"] or t["orphan"]):
        print("  FAIL: %d row(s) did not resolve their own flash"
              % (t["wrong"] + t["missing"] + t["orphan"]))
        rc = 1

    if args.compare:
        det_b, tb = scan(args.root, args.compare, samples)
        if tb["events"] == 0 or tb.get("missing_arm"):
            print("REFUSE: comparison arm read %d events" % tb["events"])
            return 1
        ka = {(r[0], r[1], r[2]): r for r in detail}
        kb = {(r[0], r[1], r[2]): r for r in det_b}
        common = set(ka) & set(kb)
        if not common:
            print("REFUSE: 0 rows joined between %s and %s -- nothing was compared"
                  % (args.arm, args.compare))
            return 1
        bad = [k for k in common if ka[k][5] != kb[k][5] or ka[k][6] != kb[k][6]]
        print("compare %s vs %s: %d rows joined, %d only in A, %d only in B, "
              "%d disagreeing (flash_time_us or flash_pe)"
              % (args.arm, args.compare, len(common),
                 len(set(ka) - set(kb)), len(set(kb) - set(ka)), len(bad)))
        for k in sorted(bad)[:10]:
            print("   %-8s evt%-8d cid=%-4d  A t=%.9g pe=%.9g   B t=%.9g pe=%.9g"
                  % (k[0], k[1], k[2], ka[k][5], ka[k][6], kb[k][5], kb[k][6]))
        if bad or set(ka) ^ set(kb):
            rc = 1

    print("VERDICT: %s" % ("PASS" if rc == 0 else "FAIL -- see above"))
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Where the PDVD PR job spends its wall clock, and how much of it is STM.

Reads the per-event `wct_pr_<run>_<evt>.log` of one arm and reports

  * the MultiAlgBlobClustering per-visitor timing table (the "MABC timing:"
    debug lines, one per pipeline stage), summed over the arm and per event;
  * the per-bundle neutrino-PR census: how many flash bundles reached
    TaggerCheckNeutrino as candidates, and how many of those selected an
    activity the STM tagger had convicted.

The second number is the size of the `nu_per_bundle_stm_only` lever (doc 25
sec 13.10): with the knob on, only STM-tagged bundles get a PR pass.  It is a
LOWER bound on the post-knob candidate count -- the knob is a selector, not
only a filter, so a bundle whose longest main is untagged but which also holds
a shorter STM-tagged main keeps a candidate, with a different identity.  The
upper bound is the STM-tagged main count (also reported).

Usage:
  python3 stm/pr_cost_census.py --tag stm3
  python3 stm/pr_cost_census.py --tag stm3 --stages    # per-stage totals only
"""
import argparse
import glob
import os
import re
from collections import OrderedDict, defaultdict

PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_TIMING = re.compile(r"MABC timing: (.+?) took ([0-9.]+) ms \(cumulative")
RE_STM1 = re.compile(r"TaggerCheckSTM: cluster (\d+) . STM=1")
RE_CAND = re.compile(r"\[nu_per_bundle\] gid (-?\d+): candidate main cluster (\d+)")
RE_SKIP_STM_ONLY = re.compile(r"not a candidate \(nu_per_bundle_stm_only\)")


def scan(log):
    """(stage -> ms, n_cand, n_cand_stm, n_stm_mains, n_skipped_by_knob)"""
    stages = OrderedDict()
    stm, cands = set(), []
    n_skip = 0
    with open(log, errors="ignore") as fp:
        for line in fp:
            m = RE_TIMING.search(line)
            if m:
                stages[m.group(1)] = stages.get(m.group(1), 0.0) + float(m.group(2))
                continue
            m = RE_STM1.search(line)
            if m:
                stm.add(int(m.group(1)))
                continue
            m = RE_CAND.search(line)
            if m:
                cands.append(int(m.group(2)))
                continue
            if RE_SKIP_STM_ONLY.search(line):
                n_skip += 1
    return stages, len(cands), sum(1 for c in cands if c in stm), len(stm), n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm tag, e.g. stm3 (work/<run>_<evt>_<tag>/)")
    ap.add_argument("--stages", action="store_true", help="print the per-stage totals only")
    a = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(PDVD, "work", "*_%s" % a.tag, "wct_pr_*.log")))
    if not logs:
        raise SystemExit("no logs under work/*_%s/" % a.tag)

    tot = defaultdict(float)
    grand = [0, 0, 0, 0]
    rows = []
    for log in logs:
        ev = os.path.basename(os.path.dirname(log))
        stages, nc, ncs, nstm, nskip = scan(log)
        for k, v in stages.items():
            tot[k] += v
        wall = sum(stages.values()) / 1000.0
        rows.append((ev, wall, nc, ncs, nstm, nskip))
        grand[0] += nc
        grand[1] += ncs
        grand[2] += nstm
        grand[3] += nskip

    if not a.stages:
        print("%-22s %9s %6s %6s %6s %7s" % ("event", "wall_s", "cands", "stmC", "stmMain", "knobSkip"))
        for ev, wall, nc, ncs, nstm, nskip in rows:
            print("%-22s %9.1f %6d %6d %6d %7d" % (ev, wall, nc, ncs, nstm, nskip))
        print("%-22s %9.1f %6d %6d %6d %7d" %
              ("TOTAL (%d events)" % len(rows), sum(r[1] for r in rows), *grand))
        if grand[0]:
            print("STM-selected fraction of bundles reaching the PR: %d/%d = %.3f"
                  % (grand[1], grand[0], grand[1] / grand[0]))
        print()

    total = sum(tot.values()) or 1.0
    print("%-38s %12s %7s" % ("MABC stage", "sum_ms", "frac"))
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print("%-38s %12.1f %6.1f%%" % (k, v, 100.0 * v / total))
    print("%-38s %12.1f" % ("TOTAL", total))


if __name__ == "__main__":
    main()

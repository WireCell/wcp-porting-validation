#!/usr/bin/env python3
"""doc sbnd_xin/pr/150 -- is the pi0 hand-scan census READABLE on a given arm?

NEW script (nothing to fork): the pi0 census resolves a hand label's gamma
through `showers[].id`, the start segment's pf_node_id.  That id is a per-event
RECONSTRUCTION index, so an arm whose clustering differs renumbers it and every
gamma reads "absent-on-arm" -- a pure numbering artefact that the census reports
in exactly the same column as a real reconstruction failure (doc pr/141 sec 3
named this for 9 gamma slots; this script measures it for a whole arm).

It prints two things per event set:
  1. the shower-id OVERLAP between two arms (Jaccard over `showers[].id`).
     Median 0 => the two arms share no ids at all, and NO id-anchored scorer can
     compare them.
  2. how many of the 132 hand-label gamma ids are actually present on each arm.

READ-ONLY.  Prints; writes nothing.

Usage:
  ./pi0_id_drift.py --a pr150s0 --b pr150csp3bw
  ./pi0_id_drift.py --a pr150s0 --b d102mpr
"""
import argparse
import csv
import importlib.util
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

_spec = importlib.util.spec_from_file_location(
    "pr126_pi0_select", os.path.join(SX, "scripts", "pr126_pi0_select.py"))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

SETS = [("98",  "em117-132denom98-manifest.tsv",  "emscan-0827"),
        ("141", "em114c-132denom141-manifest.tsv", "emscan-0828-agent5")]
OVERLAY = "pi0scan-0829-agent"


def shower_ids(tag, sample, ev):
    p = os.path.join(SX, "work-%s-%s" % (sample, tag),
                     "pr_evt%d" % ev, "calib-pr-evt%d.json" % ev)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        d = json.load(fh)
    return {int(s["id"]) for s in (d.get("showers") or ())}


def gammas(rec):
    g = ((rec or {}).get("pio") or {}).get("gammas")
    if not g or not all(x in g and (g[x].get("energy") or 0) > 0 for x in ("1", "2")):
        return None
    return [int(g[x]["shower"]) for x in ("1", "2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm tag, e.g. pr150s0")
    ap.add_argument("--b", required=True, help="arm tag, e.g. pr150csp3bw")
    x = ap.parse_args()

    ov = SEL.load_labels(OVERLAY)
    jac, both, only_a, only_b, ident = [], 0, 0, 0, 0
    lab_tot, lab_a, lab_b = 0, 0, 0
    for setname, man, tag in SETS:
        base = SEL.load_labels(tag)
        for r in csv.DictReader(open(os.path.join(SX, "em_display", man)), delimiter="\t"):
            s, ev = r["sample"], int(r["event"])
            A, B = shower_ids(x.a, s, ev), shower_ids(x.b, s, ev)
            if A is not None and B is not None:
                both += 1
                ident += 1 if A == B else 0
                jac.append(len(A & B) / max(1, len(A | B)))
            elif A is not None:
                only_a += 1
            elif B is not None:
                only_b += 1
            gs = gammas(base.get(ev)) or gammas(ov.get(ev))
            if gs:
                for gid in gs:
                    lab_tot += 1
                    lab_a += 1 if (A is not None and gid in A) else 0
                    lab_b += 1 if (B is not None and gid in B) else 0

    print("=== shower-id overlap, %s vs %s (over the 239 denominator events) ===" % (x.a, x.b))
    print("  events with a dump on both       : %d   (only %s: %d, only %s: %d)"
          % (both, x.a, only_a, x.b, only_b))
    print("  identical shower-id SET          : %d of %d" % (ident, both))
    if jac:
        print("  Jaccard of the id sets           : median %.2f  mean %.2f  max %.2f"
              % (st.median(jac), sum(jac) / len(jac), max(jac)))
    print("\n=== hand-label gamma ids found in showers[] (the census's own lookup) ===")
    print("  %-14s %3d of %d label gammas resolvable" % (x.a, lab_a, lab_tot))
    print("  %-14s %3d of %d label gammas resolvable" % (x.b, lab_b, lab_tot))
    print("  (a gamma the arm cannot resolve is reported by pr132_pi0_census.py as")
    print("   'absent-on-arm', indistinguishable there from a real missing shower)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

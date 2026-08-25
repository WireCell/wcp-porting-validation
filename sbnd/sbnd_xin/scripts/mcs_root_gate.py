#!/usr/bin/env python3
"""doc 80 -- tracking-pr.root gate for the MCS round (extends pr94_root_gate).

The pr85 hash gate compares only mabc-pr.zip and the pctree tarball; it never
opens tracking-pr.root, which is exactly the artifact doc 80 round 3 touches
(five new knob-gated kine_mcs_* T_kine branches).  A PASS from pr85 alone is
therefore vacuous for this change (doc 80 sec 10.1).  This gate compares
every tree, every branch, every entry, NaN-position-aware, and tolerates
events where a tree (e.g. T_kine on no-candidate events) is absent from BOTH
arms.

Modes:
  mcs_root_gate.py <armA> <armB>
      strict: everything identical (the knob-OFF gate).
  mcs_root_gate.py --expect-new kine_mcs_energy,kine_mcs_ambiguity,... <armA> <armB>
      armB (knob ON) may differ from armA ONLY by carrying exactly the named
      extra T_kine branches; every pre-existing tree/branch must be
      bit-identical (the knob-ON schema gate).

Exit 0 = PASS, 1 = FAIL.
"""
import argparse
import os
import re
import sys

import uproot


def norm(v):
    """Materialise a branch value; NaN -> "nan" so identical files with
    legitimate NaNs (T_rec_charge.reduced_chi2) compare equal while NaN
    POSITIONS stay enforced (pr94 convention)."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, float) and v != v:
        return "nan"
    return v


def load(path):
    out = {}
    with uproot.open(path) as f:
        for key in f.keys():
            name = key.split(";")[0]
            if name in out:
                continue          # keep the first (lowest) cycle
            t = f[key]
            try:
                arrs = t.arrays(library="np")
            except Exception as err:                       # noqa: BLE001
                out[name] = ("ERROR", str(err))
                continue
            out[name] = {b: [norm(v) for v in arrs[b]] for b in arrs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-new", default="",
                    help="comma-separated T_kine branches armB may ADD (knob-ON mode)")
    ap.add_argument("armA")
    ap.add_argument("armB")
    args = ap.parse_args()
    expect_new = set(b for b in args.expect_new.split(",") if b)

    evts = sorted(int(m.group(1)) for d in os.listdir(args.armA)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    nok = nbad = nskip = 0
    n_with_new = 0
    for e in evts:
        pa = os.path.join(args.armA, "pr_evt%d" % e, "tracking-pr.root")
        pb = os.path.join(args.armB, "pr_evt%d" % e, "tracking-pr.root")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            nskip += 1
            continue
        A, B = load(pa), load(pb)
        diffs = []
        if set(A) != set(B):
            # a tree absent from BOTH arms is fine (handled by the set
            # equality); a tree present in one arm only is always a diff --
            # the knob adds branches, never trees.
            diffs.append("trees %s vs %s" % (sorted(A), sorted(B)))
        for tname in sorted(set(A) & set(B)):
            ta, tb = A[tname], B[tname]
            if not isinstance(ta, dict) or not isinstance(tb, dict):
                diffs.append("%s unreadable" % tname)
                continue
            only_a = set(ta) - set(tb)
            only_b = set(tb) - set(ta)
            if tname == "T_kine" and expect_new:
                unexpected_b = only_b - expect_new
                if only_a or unexpected_b:
                    diffs.append("%s branches onlyA=%s unexpected_onlyB=%s"
                                 % (tname, sorted(only_a), sorted(unexpected_b)))
                if only_b & expect_new:
                    n_with_new += 1
            else:
                if only_a or only_b:
                    diffs.append("%s branches onlyA=%s onlyB=%s"
                                 % (tname, sorted(only_a), sorted(only_b)))
            for br in sorted(set(ta) & set(tb)):
                if ta[br] != tb[br]:
                    diffs.append("%s.%s" % (tname, br))
        if diffs:
            nbad += 1
            print("DIFF evt %d: %s" % (e, "; ".join(diffs[:6])))
        else:
            nok += 1
    print("# events identical%s: %d  differing: %d  skipped(no root): %d"
          % (" (mod expected-new branches)" if expect_new else "", nok, nbad, nskip))
    if expect_new:
        print("# events where armB carries the expected new branches: %d" % n_with_new)
        if n_with_new == 0:
            print("FAIL (expect-new mode but NO event carries the new branches -- vacuous)")
            return 1
    print("PASS" if nbad == 0 else "FAIL")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

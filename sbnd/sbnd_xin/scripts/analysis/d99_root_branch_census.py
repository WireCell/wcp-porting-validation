#!/usr/bin/env python3
"""doc 99 -- EXHAUSTIVE branch-level diff of tracking-pr.root between two arms.

Why not scripts/pr87_root_tree_diff.py: that one `break`s at the first differing
branch of a tree and truncates its detail list, which is right for its job
("did anything move?") and useless for this one.  The claim doc 99 has to
support is the opposite shape -- that NOTHING moved except three named columns
of one tree -- and that needs every tree, every branch, every event, with no
early exit.

Reports the complete set of differing (tree, branch) pairs with the number of
events each moved in, plus trees present on only one side.

  python3 scripts/analysis/d99_root_branch_census.py d92gatepr d99fixpr \\
      --samples ncpi0,nuecc48,mcp1k --expect T_cluster:flash_id,\\
T_cluster:flash_time_us,T_cluster:flash_pe

Exit 0 iff every differing pair is in --expect (and --expect is not required to
be exhausted -- an expected column that never moves is fine).
"""
import argparse, os, re, sys

import numpy as np
import uproot


def eq(x, y):
    a = np.asarray(x); b = np.asarray(y)
    if a.dtype == object or b.dtype == object:      # jagged / vector branches
        if len(a) != len(b):
            return False
        return all(eq(p, q) for p, q in zip(a, b))
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA"); ap.add_argument("armB")
    ap.add_argument("--samples", default="ncpi0,nuecc48,mcp1k")
    ap.add_argument("--expect", default="")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    args = ap.parse_args()
    expect = set(x.strip() for x in args.expect.split(",") if x.strip())

    diffs = {}          # "tree:branch" -> set of (sample, event)
    only = {}           # "tree" -> list of (sample, event, side)
    nev = ntree = nbr = 0
    for smp in args.samples.split(","):
        da = os.path.join(args.root, "work-%s-%s" % (smp, args.armA))
        db = os.path.join(args.root, "work-%s-%s" % (smp, args.armB))
        if not (os.path.isdir(da) and os.path.isdir(db)):
            print("skip %s: missing arm" % smp); continue
        evts = sorted(int(m.group(1)) for m in
                      (re.match(r"pr_evt(\d+)$", d) for d in os.listdir(db)) if m)
        for e in evts:
            pa = os.path.join(da, "pr_evt%d" % e, "tracking-pr.root")
            pb = os.path.join(db, "pr_evt%d" % e, "tracking-pr.root")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            nev += 1
            with uproot.open(pa) as fa, uproot.open(pb) as fb:
                ta = set(k.split(";")[0] for k in fa.keys())
                tb = set(k.split(";")[0] for k in fb.keys())
                for t in sorted(ta ^ tb):
                    only.setdefault(t, []).append((smp, e, "A" if t in ta else "B"))
                for t in sorted(ta & tb):
                    try:
                        A = fa[t].arrays(library="np")
                        B = fb[t].arrays(library="np")
                    except Exception as ex:
                        print("  read fail %s evt%d %s: %s" % (smp, e, t, ex)); continue
                    ntree += 1
                    for br in sorted(set(A) | set(B)):
                        nbr += 1
                        if br not in A or br not in B or not eq(A[br], B[br]):
                            diffs.setdefault("%s:%s" % (t, br), set()).add((smp, e))
        print("  %s done (%d events)" % (smp, len(evts)), flush=True)

    print("\ncompared %d events, %d tree instances, %d branch instances" % (nev, ntree, nbr))
    if only:
        print("\ntrees on one side only:")
        for t, v in sorted(only.items()):
            print("  %-18s %d events" % (t, len(v)))
    print("\ndiffering (tree, branch) pairs: %d" % len(diffs))
    rc = 0
    for k in sorted(diffs, key=lambda k: (-len(diffs[k]), k)):
        tag = "EXPECTED" if k in expect else "UNEXPECTED"
        if k not in expect:
            rc = 1
        print("  %-11s %-32s %d events" % (tag, k, len(diffs[k])))
    if not diffs:
        print("  (none)")
    print("\nVERDICT: %s" % ("PASS -- every difference is an expected column"
                             if rc == 0 else "FAIL -- an unexpected branch moved"))
    return rc


if __name__ == "__main__":
    sys.exit(main())

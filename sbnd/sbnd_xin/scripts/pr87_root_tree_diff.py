#!/usr/bin/env python3
"""doc 87: characterize a tracking-pr.root difference instead of just failing it.

scripts/pr94_root_gate.py answers "are these two arms' ROOT files identical?".
When save_in_scope is flipped ON the honest answer is NO -- the file gains a
T_cluster tree -- and a bare FAIL is not a useful record.  This says WHICH trees
and branches moved, so "one new tree, every existing tree identical" is a
checkable claim rather than an assertion.

NaN handling is the whole reason this is a script and not a one-liner:
T_rec_charge.reduced_chi2 carries NaNs, and a naive `!=` reports every NaN row
as a difference.  Comparisons are equal_nan=True.

Usage:
    scripts/pr87_root_tree_diff.py <armA> <armB>
Exit 0 = every tree present in BOTH arms is identical (extra trees in B are
reported, not failed); 1 = a shared tree differs.
"""
import glob, os, sys
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
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A_root, B_root = sys.argv[1], sys.argv[2]
    same = diff = skip = 0
    only_b, only_a, details = set(), set(), []
    for d in sorted(glob.glob(os.path.join(A_root, "pr_evt*"))):
        evt = os.path.basename(d)
        pa = os.path.join(d, "tracking-pr.root")
        pb = os.path.join(B_root, evt, "tracking-pr.root")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            skip += 1
            continue
        A, B = uproot.open(pa), uproot.open(pb)
        ka = {k.split(";")[0] for k in A.keys()}
        kb = {k.split(";")[0] for k in B.keys()}
        only_b |= (kb - ka); only_a |= (ka - kb)
        ok = True
        for t in sorted(ka & kb):
            ta, tb = A[t], B[t]
            if not hasattr(ta, "keys"):
                continue
            if set(ta.keys()) != set(tb.keys()):
                ok = False; details.append((evt, t, "<branch set differs>")); continue
            for br in ta.keys():
                if not eq(ta[br].array(library="np"), tb[br].array(library="np")):
                    ok = False; details.append((evt, t, br)); break
        same += ok; diff += (not ok)
    print("# events with every SHARED tree identical: %d  differing: %d  skipped: %d"
          % (same, diff, skip))
    print("# trees only in B (%s): %s" % (B_root, sorted(only_b) or "none"))
    print("# trees only in A (%s): %s" % (A_root, sorted(only_a) or "none"))
    for x in details[:12]:
        print("   DIFF %s  tree=%s  branch=%s" % x)
    print("PASS" if diff == 0 else "FAIL")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""doc pdvd/102 sec 7 -- the simulation noise floor: is the repeat arm identical to the baseline arm?

Compares every branch of every TTree in tracking-pr.root (and tracking-stm.root when present) between two arms,
event by event, recursing into jagged (per-entry vector) branches.  NaN == NaN.

Usage: d102_repeat_check.py [--dets pdvd,pdhd] [--base d102b] [--rep d102brep] [--nk 18]
"""
import argparse, os
import numpy as np
import uproot

IMG = "/home/xqian/toolkit-dev/wcp-porting-img"
RUN = {"pdvd": 900103, "pdhd": 900104}


def same(a, b):
    if isinstance(a, np.ndarray) and a.dtype != object and isinstance(b, np.ndarray) and b.dtype != object:
        if a.shape != b.shape:
            return False
        if a.dtype.kind == "f":
            return bool(np.array_equal(a, b, equal_nan=True))
        return bool(np.array_equal(a, b))
    try:
        la, lb = len(a), len(b)
    except TypeError:
        return a == b or (a != a and b != b)
    return la == lb and all(same(np.asarray(x) if isinstance(x, (list, tuple)) else x,
                                 np.asarray(y) if isinstance(y, (list, tuple)) else y) for x, y in zip(a, b))


def trees(f):
    return sorted({n.split(";")[0] for n, c in f.classnames().items() if c == "TTree"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="pdvd,pdhd")
    ap.add_argument("--base", default="d102b")
    ap.add_argument("--rep", default="d102brep")
    ap.add_argument("--nk", type=int, default=18)
    a = ap.parse_args()
    for det in a.dets.split(","):
        ok, bad = 0, []
        for k in range(a.nk):
            good = True
            for fn in ("tracking-pr.root", "tracking-stm.root"):
                pa = f"{IMG}/{det}/work/{RUN[det]}_{k}_{a.base}/{fn}"
                pb = f"{IMG}/{det}/work/{RUN[det]}_{k}_{a.rep}/{fn}"
                if not os.path.exists(pa) and not os.path.exists(pb):
                    continue
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    good = False; bad.append((k, fn, "missing")); break
                fa, fb = uproot.open(pa), uproot.open(pb)
                if trees(fa) != trees(fb):
                    good = False; bad.append((k, fn, "tree list")); break
                for t in trees(fa):
                    A = fa[t].arrays(library="np"); B = fb[t].arrays(library="np")
                    if sorted(A) != sorted(B):
                        good = False; bad.append((k, fn, t, "branches")); break
                    diff = [x for x in A if not same(A[x], B[x])]
                    if diff:
                        good = False; bad.append((k, fn, t, diff[:3])); break
                if not good:
                    break
            ok += good
        print(f"{det}: {a.rep} identical to {a.base} (all trees, all branches) on {ok}/{a.nk} events" + (f"; first differences {bad[:3]}" if bad else ""))


if __name__ == "__main__":
    main()

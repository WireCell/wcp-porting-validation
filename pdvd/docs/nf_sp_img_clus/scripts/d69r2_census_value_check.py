#!/usr/bin/env python3
"""doc pdvd/69 round 2 -- feed the values the d51g_branch_census.py change is about
to the old and the new comparison functions, directly.

    git show a5bed6a5:pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py > OLD.py
    d69r2_census_value_check.py OLD.py pdvd/docs/nf_sp_img_clus/scripts/d51g_branch_census.py

This is the negative control for the change: it corrupts exactly what the change
protects (a NaN turned into an inf, an inf that changes sign, a coordinate that
moves by less than 1e-6) and prints what each version of the comparator says.
Past gate reports cannot show this, because no past arm has a NaN or an inf in
T_stm_michel or T_stm_michel_pts (doc 69 sec 8.2).  Exit 1 if the NEW version
gets any case wrong.
"""
import importlib.util, sys
import numpy as np


def mod(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    old, new = mod(sys.argv[1], "d51g_old"), mod(sys.argv[2], "d51g_new")
    nan, inf, f32 = float("nan"), float("inf"), np.float32

    same_cases = [                     # (u, v, what the NEW same() must say)
        (nan, nan, True, "NaN vs NaN"),
        (f32(nan), f32(nan), True, "NaN vs NaN, float32"),
        (nan, inf, False, "NaN vs +inf"),
        (nan, -inf, False, "NaN vs -inf"),
        (inf, -inf, False, "+inf vs -inf"),
        (inf, inf, True, "+inf vs +inf"),
        (-inf, -inf, True, "-inf vs -inf"),
        (f32(inf), np.float64(inf), True, "+inf float32 vs float64"),
        (nan, 1.0, False, "NaN vs 1.0"),
        (inf, 1e308, False, "+inf vs 1e308"),
        (1.0, 1.0 + 1e-9, False, "1.0 vs 1.0+1e-9"),
        (0.0, -0.0, True, "0.0 vs -0.0 (unchanged by design)"),
        (np.int32(3), np.int32(3), True, "int 3 vs 3"),
        (np.int32(3), np.int32(4), False, "int 3 vs 4"),
    ]
    bad = 0
    print("%-34s %6s %6s %6s" % ("same(u, v)", "old", "new", "want"))
    for u, v, want, what in same_cases:
        o, n = old.same(u, v), new.same(u, v)
        bad += n != want
        print("%-34s %6s %6s %6s%s" % (what, o, n, want, "" if n == want else "   <-- WRONG"))

    pts_cases = [                      # the old key was round(float(x), 6)
        (1.0000001, 1.0000004, False, "1.0000001 vs 1.0000004"),
        (123.4567891, 123.4567894, False, "123.4567891 vs 123.4567894"),
        (nan, nan, True, "NaN vs NaN"),
        (nan, inf, False, "NaN vs +inf"),
        (2.5, 2.5, True, "2.5 vs 2.5"),
    ]
    print("\n%-34s %6s %6s %6s" % ("pts key equal", "old", "new", "want"))
    for u, v, want, what in pts_cases:
        o = round(float(u), 6) == round(float(v), 6)
        n = new.exact(u) == new.exact(v)
        bad += n != want
        print("%-34s %6s %6s %6s%s" % (what, o, n, want, "" if n == want else "   <-- WRONG"))

    a = sorted([(0, new.exact(nan), new.exact(1.0)), (0, new.exact(2.0), new.exact(3.0))])
    b = sorted([(0, new.exact(2.0), new.exact(3.0)), (0, new.exact(nan), new.exact(1.0))])
    print("\nNaN-bearing point lists sort to the same order: %s" % (a == b))
    bad += a != b
    print("RESULT: %s" % ("all cases as intended" if not bad else "%d WRONG" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

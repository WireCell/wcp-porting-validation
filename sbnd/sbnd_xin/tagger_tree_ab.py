#!/usr/bin/env python3
"""Generic T_tagger branch-by-branch diff between two tracking-*.root arms.

Usage: tagger_tree_ab.py <A.root> <B.root> [tree]

Prints the branch count, how many are identical, and every branch that moved
with its A/B values.  Same comparison core as ssm_tagger_ab.py (doc pr/2
sec 2e(i-a)) but with no knob-specific identity check, so it works for any
two arms of the same event.
"""
import sys
import numpy as np
import uproot

A, B = sys.argv[1], sys.argv[2]
TREE = sys.argv[3] if len(sys.argv) > 3 else "T_tagger"

ta = uproot.open(A)[TREE]
tb = uproot.open(B)[TREE]
ka, kb = sorted(ta.keys()), sorted(tb.keys())
if ka != kb:
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    print(f"branch sets differ: only-A={only_a}  only-B={only_b}")
    sys.exit(1)

da = ta.arrays(ka, library="np")
db = tb.arrays(kb, library="np")

moved, same = [], 0
for k in ka:
    va, vb = np.asarray(da[k]), np.asarray(db[k])
    if va.dtype == object:  # jagged branch: compare entry by entry
        eq = len(va) == len(vb) and all(np.array_equal(x, y) for x, y in zip(va, vb))
    else:
        eq = va.shape == vb.shape and np.array_equal(va, vb)
    if eq:
        same += 1
    else:
        moved.append(k)

print(f"tree {TREE}: branches {len(ka)}   identical {same}   moved {len(moved)}")
for k in moved:
    print(f"  MOVED {k:38s} A={np.asarray(da[k])[0]!r}  B={np.asarray(db[k])[0]!r}")

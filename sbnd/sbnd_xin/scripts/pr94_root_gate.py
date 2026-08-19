#!/usr/bin/env python3
"""doc pr/94 -- per-branch VALUE gate between two arms' tracking-pr.root.

The pr85 hash gate covers mabc-pr.zip and the pctree tarball but not the ROOT
file, which is exactly where doc pr/94 changes the schema.  This compares every
tree, every branch, every entry.

Usage: pr94_root_gate.py <armA> <armB>
Exit 0 = identical everywhere; 1 otherwise.
"""
import os
import re
import sys

import uproot


def norm(v):
    """Fully materialise a branch value into plain Python so == is unambiguous.

    NaN is mapped to the string "nan" on purpose.  T_rec_charge.reduced_chi2
    legitimately carries NaN (2 of 962 entries on nueCC48 evt 389538, a
    pre-existing property of the fit, unrelated to doc pr/94), and NaN != NaN
    would otherwise make an identical file compare as different.  Bit-identity
    of the NaN POSITIONS is still enforced, which is what we actually want.
    """
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
    a_root, b_root = sys.argv[1], sys.argv[2]
    evts = sorted(int(m.group(1)) for d in os.listdir(a_root)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    nok = nbad = nskip = 0
    for e in evts:
        pa = os.path.join(a_root, "pr_evt%d" % e, "tracking-pr.root")
        pb = os.path.join(b_root, "pr_evt%d" % e, "tracking-pr.root")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            nskip += 1
            continue
        A, B = load(pa), load(pb)
        diffs = []
        if set(A) != set(B):
            diffs.append("trees %s vs %s" % (sorted(A), sorted(B)))
        for tname in sorted(set(A) & set(B)):
            ta, tb = A[tname], B[tname]
            if not isinstance(ta, dict) or not isinstance(tb, dict):
                diffs.append("%s unreadable" % tname)
                continue
            only_a, only_b = set(ta) - set(tb), set(tb) - set(ta)
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
    print("# events identical: %d  differing: %d  skipped(no root): %d"
          % (nok, nbad, nskip))
    print("PASS" if nbad == 0 else "FAIL")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

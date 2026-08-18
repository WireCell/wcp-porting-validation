#!/usr/bin/env python3
"""pr/84 round 2 -- display-only gate between two PR arms.

For every event in <new_arm>, requires (member-content hashes via
abtest/hash_archive.py, NOT raw cmp -- M2):
  - pctree-pr-evt<ID>.tar.gz identical;
  - every member of mabc-pr.zip identical EXCEPT the one ending in
    "0-mc.json", which MAY differ (the pr/34 precedent: a display-only PF
    knob moves ONLY the particle-flow tree).

Prints one line per event whose mc.json moved (the census input) and FAILs
if any OTHER member moved.

Usage:
  pr84r2_disp_gate.py <new_arm> <ref_arm> [<new_arm2> <ref_arm2> ...]
"""
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "abtest"))
from hash_archive import members  # noqa: E402


def member_hashes(path):
    out = {}
    for name, payload in members(path):
        out[name] = hashlib.sha256(payload).hexdigest()
    return out


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)
    n_pass = n_fail = n_moved = 0
    for new_arm, ref_arm in zip(args[::2], args[1::2]):
        for d in sorted(glob.glob(os.path.join(new_arm, "pr_evt*"))):
            evt = os.path.basename(d).replace("pr_evt", "")
            ok = True
            # pctree must be identical
            f = f"pctree-pr-evt{evt}.tar.gz"
            a, b = os.path.join(d, f), os.path.join(ref_arm, f"pr_evt{evt}", f)
            if not (os.path.exists(a) and os.path.exists(b)):
                print(f"FAIL missing evt={evt} f={f}")
                ok = False
            else:
                ha = member_hashes(a)
                hb = member_hashes(b)
                if ha != hb:
                    print(f"FAIL pctree moved evt={evt}")
                    ok = False
            # mabc: only 0-mc.json may differ
            a = os.path.join(d, "mabc-pr.zip")
            b = os.path.join(ref_arm, f"pr_evt{evt}", "mabc-pr.zip")
            if not (os.path.exists(a) and os.path.exists(b)):
                print(f"FAIL missing evt={evt} f=mabc-pr.zip")
                ok = False
            else:
                ha = member_hashes(a)
                hb = member_hashes(b)
                if set(ha) != set(hb):
                    print(f"FAIL member set changed evt={evt}: "
                          f"{sorted(set(ha) ^ set(hb))}")
                    ok = False
                else:
                    for name in sorted(ha):
                        if ha[name] == hb[name]:
                            continue
                        if name.endswith("0-mc.json"):
                            n_moved += 1
                            print(f"MOVED evt={evt} member={name}")
                        else:
                            print(f"FAIL non-mc member moved evt={evt} "
                                  f"member={name}")
                            ok = False
            if ok:
                n_pass += 1
            else:
                n_fail += 1
    print(f"# disp gate: PASS={n_pass} FAIL={n_fail} mc.json-moved={n_moved}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""pr/83 round 3 -- byte-identity gate between a rerun PR arm and its
reference production arm, over every event present in the rerun arm.

Compares the rollup member-content hash (abtest/hash_archive.py logic,
imported, NOT raw cmp -- M2) of mabc-pr.zip and pctree-pr-evt<ID>.tar.gz.

Usage:
  pr83r3_hash_gate.py <new_arm> <ref_arm> [<new_arm2> <ref_arm2> ...]

Exit 0 iff every compared archive matches (a missing reference event is a
FAIL, not a skip).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "abtest"))
from hash_archive import members  # noqa: E402


def rollup(path):
    import hashlib
    h = hashlib.sha256()
    try:
        for name, payload in members(path):
            h.update(name.encode())
            h.update(hashlib.sha256(payload).digest())
    except Exception as e:
        return f"ERROR:{e}"
    return h.hexdigest()


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)
    n_pass = n_fail = 0
    for new_arm, ref_arm in zip(args[::2], args[1::2]):
        for d in sorted(glob.glob(os.path.join(new_arm, "pr_evt*"))):
            evt = os.path.basename(d).replace("pr_evt", "")
            for f in ("mabc-pr.zip", f"pctree-pr-evt{evt}.tar.gz"):
                a = os.path.join(d, f)
                b = os.path.join(ref_arm, f"pr_evt{evt}", f)
                if not os.path.exists(a) or not os.path.exists(b):
                    n_fail += 1
                    print(f"FAIL missing evt={evt} f={f} "
                          f"(new={os.path.exists(a)} ref={os.path.exists(b)})")
                    continue
                ha, hb = rollup(a), rollup(b)
                if ha == hb and not ha.startswith("ERROR"):
                    n_pass += 1
                else:
                    n_fail += 1
                    print(f"FAIL evt={evt} f={f} new={ha[:16]} ref={hb[:16]}")
    print(f"# gate: PASS={n_pass} FAIL={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Doc 60 A/B gate: compare the PR-tagger products of two work-mcp1kall arms.

Both arms run the SAME production flag set over the SAME Q/L pctrees (the arms'
ql_evt<ID>/pctree-evt<ID>.tar.gz are symlinks to one set of doc-59 tarballs), so
the only difference is the binary.  Archives are compared by MEMBER CONTENT
(abtest/hash_archive.py) because tar/zip embed mtimes (M2); the per-bundle table
is plain text and is hashed directly.  tracking-stm.root is skipped: ROOT files
carry a creation timestamp and are not byte-reproducible.

Usage:
  ./d60_ab_report.py <arm-a-root> <arm-b-root> [--verbose]

Repro (doc 60 §6):
  ./d60_ab_report.py work-mcp1kall-d60base work-mcp1kall-d60fix
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
                                "..", "..", "abtest"))
from hash_archive import members  # noqa: E402

ARCHIVES = ("mabc-pr.zip", "pctree-pr-evt{id}.tar.gz")
TEXT = ("nusel-evt{id}.tsv",)


def rollup(path):
    """Same member-content rollup hash_archive.py prints (kept in lockstep)."""
    roll = hashlib.sha256()
    for name, payload in members(path):
        roll.update(hashlib.sha256(name.encode() + payload).hexdigest().encode())
    return roll.hexdigest()


def digest(path):
    """Content hash, or a marker string for missing / empty / unreadable."""
    if not os.path.exists(path):
        return "MISSING"
    if os.path.getsize(path) == 0:
        return "EMPTY"
    try:
        if path.endswith((".zip", ".tar.gz")):
            return rollup(path)
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as exc:                      # truncated archive, bad zip
        return "UNREADABLE:%s" % type(exc).__name__


def status_rc(root, entry):
    p = os.path.join(root, ".status", str(entry))
    if not os.path.exists(p):
        return None
    head = open(p).read().split()
    return head[0].split("=", 1)[1] if head else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "--verbose" in sys.argv[1:]
    # --archives-only: compare just the two WCT archives.  Needed when an arm
    # was built from a pctree-only symlink farm, so nusel_extract.py could not
    # run (it also wants the Q/L step's mabc-all-apa.zip / calib json) and the
    # per-bundle tsv does not exist.  The archives are the tagger's actual
    # output; the tsv is derived from them.
    checks = ARCHIVES if "--archives-only" in sys.argv[1:] else ARCHIVES + TEXT
    if len(args) != 2:
        sys.exit(__doc__)
    a_root, b_root = args

    events = sorted(d[len("nusel_evt"):] for d in os.listdir(a_root)
                    if d.startswith("nusel_evt"))
    b_events = set(d[len("nusel_evt"):] for d in os.listdir(b_root)
                   if d.startswith("nusel_evt"))

    same, diff, only_a, only_b = [], [], [], []
    for eid in events:
        if eid not in b_events:
            only_a.append(eid)
            continue
        deltas = []
        for pat in checks:
            name = pat.format(id=eid)
            ha = digest(os.path.join(a_root, "nusel_evt" + eid, name))
            hb = digest(os.path.join(b_root, "nusel_evt" + eid, name))
            if ha != hb:
                deltas.append((name, ha, hb))
        (diff if deltas else same).append((eid, deltas))
    only_b = sorted(b_events - set(events))

    print("arm A: %s" % a_root)
    print("arm B: %s" % b_root)
    print("events compared : %d" % (len(same) + len(diff)))
    print("  identical     : %d" % len(same))
    print("  DIFFERENT     : %d" % len(diff))
    print("  only in A     : %d %s" % (len(only_a), only_a[:10] or ""))
    print("  only in B     : %d %s" % (len(only_b), only_b[:10] or ""))

    for eid, deltas in diff:
        print("\nevt %s  rc A=%s B=%s" % (eid, status_rc(a_root, eid),
                                          status_rc(b_root, eid)))
        for name, ha, hb in deltas:
            print("    %-28s A=%-16s B=%s" % (name, ha[:16], hb[:16]))

    if verbose and same:
        print("\nfirst 5 identical events:")
        for eid, _ in same[:5]:
            print("  %s %s" % (eid, digest(os.path.join(
                a_root, "nusel_evt" + eid, "mabc-pr.zip"))[:16]))

    print("\n%s" % ("PASS: byte-identical on every compared event"
                    if not diff else
                    "DIFF: %d event(s) differ (see above)" % len(diff)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

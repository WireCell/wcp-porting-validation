#!/usr/bin/env python3
"""doc 81 sec 10 -- re-check a frozen stage-A manifest against the SURVIVING arm.

The freeze is only worth something if it can actually be re-checked after the
reference arms are gone.  This is that check, run BEFORE the deletion so the
answer is known rather than hoped for: recompute every rollup from
``work-<s>-grp0825`` and diff it against
``state-20260825b/hashes/stagea-<s>.tsv``, which was computed from
``work-img-<s>`` + ``work-<s>-ql0819``.

An all-match is doc 81 sec 7's 24536/24536 gate reproduced today, from the
frozen file, on the arm that survives -- i.e. proof that retiring the reference
side does not cost the claim.  It is also the exact command a future round
would run to re-verify it.

The frozen file's own product labels drive the comparison (``img/<base>`` ->
``evt<N>/<base>.npz``, ``ql/<f>`` -> ``ql_evt<N>/<f>``), so this cannot silently
check a different product set than the one that was frozen.

usage: verify_frozen_stagea_20260825b.py <sample> [<sample> ...]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from hash_manifest_stagea_20260825b import rollup  # noqa: E402

HASHDIR = os.path.join(HERE, "state-20260825b", "hashes")


def path_for(arm, evt, label):
    kind, name = label.split("/", 1)
    if kind == "img":
        return os.path.join(arm, "evt%d" % evt, name + ".npz")
    return os.path.join(arm, "ql_evt%d" % evt, name)


def one(job):
    evt, label, want_h, want_n, arm = job
    got_h, got_n = rollup(path_for(arm, evt, label))
    if got_h is None:
        return evt, label, "MISSING", want_n, 0
    if got_h != want_h:
        return evt, label, "DIFFER", want_n, got_n
    return evt, label, "SAME", want_n, got_n


def main():
    samples = sys.argv[1:]
    if not samples:
        sys.exit("usage: verify_frozen_stagea_20260825b.py <sample> [<sample> ...]")
    os.chdir(ROOT)
    grand_same = grand_bad = 0
    for s in samples:
        arm = "work-%s-grp0825" % s
        man = os.path.join(HASHDIR, "stagea-%s.tsv" % s)
        if not os.path.isdir(arm):
            sys.exit("!! no such surviving arm: %s" % arm)
        if not os.path.exists(man):
            sys.exit("!! no frozen manifest: %s" % man)
        jobs = []
        for line in open(man):
            if line.startswith("#"):
                continue
            evt, label, h, n = line.rstrip("\n").split("\t")
            jobs.append((int(evt), label, h, int(n), arm))
        same = bad = 0
        first = []
        with ProcessPoolExecutor(max_workers=10) as ex:
            for evt, label, verdict, want_n, got_n in ex.map(one, jobs, chunksize=16):
                if verdict == "SAME":
                    same += 1
                else:
                    bad += 1
                    if len(first) < 5:
                        first.append("evt%d %s %s (%d vs %d members)"
                                     % (evt, label, verdict, want_n, got_n))
        grand_same += same
        grand_bad += bad
        status = "PASS" if not bad else "FAIL"
        print("  %-8s %s  %5d/%5d rollups identical to the frozen reference"
              % (s, status, same, same + bad))
        for f in first:
            print("        !! " + f)
    print("\n%s -- %d/%d frozen stage-A rollups reproduced from the surviving "
          "grp0825 arms" % ("PASS" if not grand_bad else "FAIL",
                            grand_same, grand_same + grand_bad))
    return 0 if not grand_bad else 1


if __name__ == "__main__":
    sys.exit(main())

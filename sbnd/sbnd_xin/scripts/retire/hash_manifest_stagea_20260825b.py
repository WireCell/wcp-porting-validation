#!/usr/bin/env python3
"""doc 81 sec 10 -- freeze the STAGE-A reference side before it is retired.

Companion to ``hash_manifest_20260825.py``, which freezes PR arms.  It cannot
be reused here and the failure would be SILENT: its ``EVT_RE`` is
``r"pr_evt(\\d+)$"``, and stage-A arms are laid out as ``evt<N>/`` (imaging) and
``ql_evt<N>/`` (Q/L).  Every subdir would be skipped, the arm would get a
header-only .tsv, and a ``[ -s "$f" ]`` interlock would happily pass it -- an
M1-shaped vacuous PASS with 29 G deleted behind it.  Hence a second tool, and
hence interlock 4 in ``retire_20260825b.sh`` asserts the ROW COUNT, not that
the file is non-empty.

WHAT IS BEING PRESERVED.  Doc 81 sec 7 gated ``work-<s>-grp0825`` against
``work-img-<s>`` + ``work-<s>-ql0819`` and got 24536/24536 archives
member-content identical.  This round retires the reference side of that gate,
so without a freeze the PASS stops being re-checkable the day it is deleted --
the same reasoning, and the same remedy, as interlock 4 of the 08-25 round.

The product set is taken from ``scripts/multi/stagea_gate.py`` (NPZ + QL +
pctree, 8 per event) by IMPORT, not by re-listing it here, so the frozen
manifest cannot drift from the definition the gate actually used.

Member content, never container bytes: zip/tar embed mtimes (CLAUDE.md M2).
That is not academic here -- ncpi0 evt105946's ``icluster-apa0-active.npz`` is
1418630 bytes in ``work-img-ncpi0`` and 5760756 bytes in
``work-ncpi0-grp0825``, and the two are nonetheless member-for-member
identical; only the container's compression differs.

Reads only.  Never moves, deletes or rewrites anything under work-*.

usage: hash_manifest_stagea_20260825b.py <sample> [<sample> ...]
       (sample = nuecc48 | ncpi0 | mcp1k | mcp2k; writes
        state-20260825b/hashes/stagea-<sample>.tsv)
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "multi"))
from stagea_gate import NPZ, QL, members  # noqa: E402

OUT = os.path.join(HERE, "state-20260825b", "hashes")


def rollup(path):
    """(sha256 over member_name+payload, n_members) -- pr85_hash_gate's shape."""
    m = members(path)
    if m is None:
        return None, 0
    h = hashlib.sha256()
    for name in sorted(m):
        h.update(name.encode())
        h.update(m[name].encode())
    return h.hexdigest(), len(m)


def one(job):
    evt, label, path = job
    h, n = rollup(path)
    return evt, label, h, n


def main():
    samples = sys.argv[1:]
    if not samples:
        sys.exit(__doc__.strip().splitlines()[-3])
    os.makedirs(OUT, exist_ok=True)
    os.chdir(ROOT)
    rc = 0
    for s in samples:
        img = "work-img-%s" % s
        ql = "work-%s-ql0819" % s
        for d in (img, ql):
            if not os.path.isdir(d):
                sys.exit("!! no such arm: %s" % d)
        # The event set is the Q/L arm's ql_evt<N> dirs -- that is what stage A
        # actually produced.  ql0819's own evt<N> entries are SYMLINKS into the
        # img arm, so listing them here would double-count.
        evts = sorted(int(d[len("ql_evt"):]) for d in os.listdir(ql)
                      if d.startswith("ql_evt") and d[len("ql_evt"):].isdigit())
        jobs = []
        for evt in evts:
            for base in NPZ:
                jobs.append((evt, "img/" + base,
                             os.path.join(img, "evt%d" % evt, base + ".npz")))
            for f in QL + ["pctree-evt%d.tar.gz" % evt]:
                jobs.append((evt, "ql/" + f,
                             os.path.join(ql, "ql_evt%d" % evt, f)))
        rows, missing = [], []
        with ProcessPoolExecutor(max_workers=10) as ex:
            for evt, label, h, n in ex.map(one, jobs, chunksize=16):
                if h is None:
                    missing.append((evt, label))
                    continue
                rows.append((evt, label, h, n))
        rows.sort()
        dest = os.path.join(OUT, "stagea-%s.tsv" % s)
        with open(dest, "w") as fp:
            fp.write("# doc 81 sec 10 -- stage-A reference-side member-content hashes\n")
            fp.write("# the reference half of doc 81 sec 7's 24536/24536 gate\n")
            fp.write("# img=%s ql=%s events=%d products=%d missing=%d\n"
                     % (img, ql, len(evts), len(rows), len(missing)))
            fp.write("#evt\tproduct\trollup_sha256\tmembers\n")
            for evt, label, h, n in rows:
                fp.write("%d\t%s\t%s\t%d\n" % (evt, label, h, n))
        flag = ""
        if missing:
            flag = "  !! %d MISSING (e.g. evt%d %s)" % (
                len(missing), missing[0][0], missing[0][1])
            rc = 1
        print("stagea-%-9s %4d events %6d products -> %s%s"
              % (s, len(evts), len(rows), dest, flag))
    return rc


if __name__ == "__main__":
    sys.exit(main())

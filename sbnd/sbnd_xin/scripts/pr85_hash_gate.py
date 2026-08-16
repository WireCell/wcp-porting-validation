#!/usr/bin/env python3
"""doc pr/85 -- byte-identity gate between two PR arms.

For every event present in arm A, compares the member-content hash (the
abtest/hash_archive.py rollup: sha256 over member_name + payload, members
sorted by name) of the per-event archives in arm B.  Only the hash and
member count are compared -- NEVER the printed path (the pr/83 trap).

Usage: pr85_hash_gate.py <armA> <armB> [--archives mabc-pr.zip,pctree]
Exit 0 = every archive of every common event matches; 1 otherwise.
"""
import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest")
import hash_archive  # noqa: E402


def rollup(path):
    """The hash_archive.py main() rollup: (sha256-of-member-hashes, count)."""
    r = hashlib.sha256()
    n = 0
    for name, payload in hash_archive.members(path):
        h = hashlib.sha256(name.encode() + payload).hexdigest()
        r.update(h.encode())
        n += 1
    return r.hexdigest(), n


def archives_of(prdir, evt):
    out = []
    for name in ("mabc-pr.zip", "pctree-pr-evt%d.tar.gz" % evt):
        p = os.path.join(prdir, name)
        if os.path.exists(p):
            out.append(p)
    return out


def one(job):
    evt, pa, pb = job
    return evt, os.path.basename(pa), rollup(pa), rollup(pb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armA")
    ap.add_argument("armB")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    evts = sorted(int(m.group(1)) for d in os.listdir(args.armA)
                  if (m := re.match(r"pr_evt(\d+)$", d)))
    jobs, missing = [], []
    for e in evts:
        da = os.path.join(args.armA, "pr_evt%d" % e)
        db = os.path.join(args.armB, "pr_evt%d" % e)
        if not os.path.isdir(db):
            missing.append(e)
            continue
        aa = archives_of(da, e)
        bb = archives_of(db, e)
        an = {os.path.basename(p) for p in aa}
        bn = {os.path.basename(p) for p in bb}
        if an != bn:
            missing.append(e)
            continue
        for p in aa:
            jobs.append((e, p, os.path.join(db, os.path.basename(p))))

    fails = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for evt, name, (ha, na), (hb, nb) in ex.map(one, jobs, chunksize=4):
            if (ha, na) != (hb, nb):
                fails.append((evt, name))

    print("# events in A: %d  compared archives: %d  missing/unpaired events: %d"
          % (len(evts), len(jobs), len(missing)))
    if missing:
        print("# missing:", " ".join(str(e) for e in missing))
    if fails:
        print("FAIL %d archives differ:" % len(fails))
        for evt, name in fails:
            print("  evt %d %s" % (evt, name))
        sys.exit(1)
    print("PASS all %d archives byte-identical" % len(jobs))


if __name__ == "__main__":
    main()

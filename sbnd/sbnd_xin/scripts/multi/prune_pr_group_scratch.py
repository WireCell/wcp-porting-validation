#!/usr/bin/env python3
"""doc 81 round 5 -- drop the stage-B group scratch, but only where it is
PROVEN to be a duplicate of the stage-A Q/L pctrees it was assembled from.

`run_pr_chain_batch.sh:1692` builds `<pr_root>/.groups/g<N>.tar.gz` by merging
the per-event `<ql_root>/ql_evt<ID>/pctree-evt<ID>.tar.gz` of that group -- it is
the group job's INPUT, staged for one wire-cell process, not a product.  It is
the stage-B twin of the stage-A group scratch `prune_group_scratch.sh` already
reclaims, and nothing reads it after the job: the gates
(pr85_hash_gate.py / pr94_root_gate.py / nusel_extract.py) all work off the
per-event `pr_evt<ID>/` outputs.

This does NOT delete on that reasoning alone.  For each `g<N>.tar.gz` it
reconstructs the member->sha256 map of the corresponding `ql_evt<ID>` pctrees
and requires the two to be EXACTLY equal -- same member names, same payload
hashes, nothing extra on either side -- before unlinking.  Anything that fails,
or whose Q/L side is missing, is kept and reported.  So a group tar that is not
what this docstring claims survives the run.

usage: prune_pr_group_scratch.py [--apply] <pr_root> <ql_root> [<pr_root> <ql_root> ...]
       (default is a dry run)
"""
import hashlib, os, re, sys, tarfile
from concurrent.futures import ProcessPoolExecutor

EVT = re.compile(r"_(\d+)_")


def digest(path):
    out = {}
    with tarfile.open(path) as t:
        for ti in t:
            if ti.isfile():
                out[ti.name] = hashlib.sha256(t.extractfile(ti).read()).hexdigest()
    return out


def check(job):
    gtar, ql_root = job
    try:
        gm = digest(gtar)
        evts = sorted({m.group(1) for n in gm for m in [EVT.search(n)] if m})
        if not evts:
            return (gtar, "keep:no-event-keys", 0)
        qm = {}
        for e in evts:
            p = os.path.join(ql_root, "ql_evt%s" % e, "pctree-evt%s.tar.gz" % e)
            if not os.path.exists(p):
                return (gtar, "keep:missing-ql-%s" % e, 0)
            qm.update(digest(p))
        if gm != qm:
            extra_g, extra_q = len(set(gm) - set(qm)), len(set(qm) - set(gm))
            bad = sum(1 for n in set(gm) & set(qm) if gm[n] != qm[n])
            return (gtar, "keep:NOT-duplicate(+g%d,+q%d,mismatch%d)" % (extra_g, extra_q, bad), 0)
        return (gtar, "duplicate", os.path.getsize(gtar))
    except Exception as e:                                   # noqa: BLE001
        return (gtar, "keep:error-%s" % type(e).__name__, 0)


def main():
    args = sys.argv[1:]
    apply_ = "--apply" in args
    args = [a for a in args if a != "--apply"]
    if not args or len(args) % 2:
        raise SystemExit(__doc__)

    jobs = []
    for pr_root, ql_root in zip(args[0::2], args[1::2]):
        gdir = os.path.join(pr_root, ".groups")
        if not os.path.isdir(gdir):
            print("no .groups in %s" % pr_root)
            continue
        for f in sorted(os.listdir(gdir)):
            if re.fullmatch(r"g\d+\.tar\.gz", f):
                jobs.append((os.path.join(gdir, f), ql_root))
    if not jobs:
        raise SystemExit("nothing to check")

    dup = kept = 0
    freed = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for gtar, verdict, size in ex.map(check, jobs):
            if verdict == "duplicate":
                dup += 1
                freed += size
                if apply_:
                    os.unlink(gtar)
            else:
                kept += 1
                print("  KEEP %s: %s" % (gtar, verdict))
    print("%s: %d/%d verified duplicates, %.2f GiB%s; %d kept"
          % ("REMOVED" if apply_ else "DRY RUN", dup, len(jobs),
             freed / 2**30, "" if apply_ else " would be freed", kept))
    return 0


if __name__ == "__main__":
    sys.exit(main())

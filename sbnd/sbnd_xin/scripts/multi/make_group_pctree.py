#!/usr/bin/env python3
"""Concatenate per-event pctree archives into ONE group archive, and emit the
group's RSE map.

doc 76 round 2.  A group of events is fed to a single wire-cell process as one
tensor-file archive.  TensorFileSource finds event boundaries by watching the
ident change as it reads the stream, so MEMBER ORDER MATTERS: each event's
members must stay contiguous and in the order TensorFileSink wrote them
(tensorset metadata first, then its tensors).  Building the archive with a
sorted `tar czf *` instead splits every event in two and the source re-emits
each ident twice -- which is exactly what happened the first time this was
tried by hand, and it silently produced truncated second copies rather than an
error.  Hence this script rather than a shell one-liner.

The RSE map is read from each event's staged opflash tensor-set metadata, the
same source run_pr_chain_batch.sh uses per event.  It is needed because a group
can span many runs (SBND nueCC48 spans 12), while the job's run/subrun TLAs are
one pair for the whole process.

usage:
  make_group_pctree.py --ql-root DIR --out ARCHIVE [--rse-map JSON] EVT [EVT...]
"""
import argparse, json, os, sys, tarfile


def event_pctree(ql_root, evt):
    return os.path.join(ql_root, "ql_evt%s" % evt, "pctree-evt%s.tar.gz" % evt)


def read_rse(ql_root, evt):
    """(run, subrun) from the event's staged opflash metadata, or None."""
    path = os.path.join(ql_root, "ql_evt%s" % evt, "opflash_apa0.tar.gz")
    if not os.path.exists(path):
        return None
    member = "opflash_tensorset_%s_metadata.json" % evt
    try:
        with tarfile.open(path, "r:*") as tf:
            fp = tf.extractfile(member)
            if fp is None:
                return None
            md = json.loads(fp.read())
    except (KeyError, OSError, ValueError):
        return None
    if "run" not in md:
        return None
    return int(md.get("run", 0)), int(md.get("subrun", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ql-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rse-map")
    ap.add_argument("events", nargs="+")
    args = ap.parse_args()

    missing = [e for e in args.events if not os.path.exists(event_pctree(args.ql_root, e))]
    if missing:
        sys.exit("no pctree for event(s): %s" % " ".join(missing))

    mode = "w:gz" if args.out.endswith(".gz") else "w"
    n = 0
    with tarfile.open(args.out, mode) as out:
        for evt in args.events:
            with tarfile.open(event_pctree(args.ql_root, evt), "r:*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    out.addfile(m, tf.extractfile(m))
                    n += 1
    # doc 81: record the Q/L root this group was built from.  ASSERT 8 of the
# retirement plan re-derives PR-arm provenance from evidence on disk, and a
# group job's per-event logs never name the ql_evt paths a per-event job logs.
    print("group provenance: ql_root=%s" % os.path.abspath(args.ql_root))
    print("%s: %d events, %d members" % (args.out, len(args.events), n))

    if args.rse_map:
        rse = {}
        for evt in args.events:
            got = read_rse(args.ql_root, evt)
            if got is not None:
                rse[str(evt)] = [got[0], got[1]]
        with open(args.rse_map, "w") as fp:
            json.dump(rse, fp)
        print("%s: %d/%d events with run/subrun" % (args.rse_map, len(rse), len(args.events)))


if __name__ == "__main__":
    main()

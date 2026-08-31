#!/usr/bin/env python3
"""{ "<event id>": [run, subrun] } for every event in a staged opflash archive.

doc 76 round 2.  A group of events runs in one wire-cell process, but the job's
run/subrun TLAs are a single pair -- and an SBND group can span many runs (the
nueCC48 sample spans 12).  MultiAlgBlobClustering's `rse_map` fixes that per
event; this builds it from the same metadata the per-event runners read.

usage: reco1_rse_map.py --opflash ARCHIVE --out JSON
"""
import argparse
import json
import re
import sys
import tarfile

MEMBER = re.compile(r"opflash_tensorset_(\d+)_metadata\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opflash", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rse = {}
    with tarfile.open(args.opflash, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            hit = MEMBER.search(m.name)
            if not hit:
                continue
            md = json.loads(tf.extractfile(m).read())
            if "run" not in md:
                continue
            rse[hit.group(1)] = [int(md.get("run", 0)), int(md.get("subrun", 0))]

    with open(args.out, "w") as fp:
        json.dump(rse, fp)
    print("%s: %d events" % (args.out, len(rse)))
    if not rse:
        print("WARNING: no run/subrun metadata in %s -- the job will fall back "
              "to its configured pair for every event" % args.opflash, file=sys.stderr)


if __name__ == "__main__":
    main()

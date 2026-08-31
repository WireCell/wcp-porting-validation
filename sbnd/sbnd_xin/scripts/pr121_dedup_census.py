#!/usr/bin/env python3
"""doc pr/121 -- examine_shower_1 dedup census: every EX1_DEDUP probe fire
(NeutrinoShowerClustering.cxx accept-branch dedup) across the given dbg arms,
classed by the erased shower's segment count.

The dedup was written for stale SINGLE-segment wrappers left by
shower_clustering_with_nv_in_main_cluster, but its predicate never checks the
segment count: on SBND 17394-348471 it erased a 13-member 352.6 MeV shower
(retargeted onto proton seg 12052 by examine_showers_retarget_seed) with no
re-homing, orphaning 12 EM segments from PF output (doc pr/115 sec 17.7).
A multi-segment erase is the orphan-risk class the shower_ex1_dedup_rehome
knob targets; this census counts how often each class fires and what the
event-level ownership damage is (calib dump: segments with shower_id == -1).

Repro:
  ./scripts/pr121_dedup_census.py --tsv docs/pr/pr121-dedup-census.tsv \
      'work-pr121r1-dbgA-*' 'work-pr121r1-dbg141-*'
"""
import argparse
import glob
import json
import os
import re

KV = re.compile(r"(\w+)=([^\s()]+)")


def parse_kv(line):
    d = {}
    for k, v in KV.findall(line):
        try:
            d[k] = float(v) if "." in v else int(v)
        except ValueError:
            d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--tsv")
    args = ap.parse_args()

    rows = []
    for g in args.globs:
        for root in sorted(glob.glob(g)):
            for log in sorted(glob.glob(os.path.join(root, "pr_evt*", "stdout.log"))):
                ev = int(os.path.basename(os.path.dirname(log))[len("pr_evt"):])
                dedups = []
                for line in open(log, errors="replace"):
                    if line.startswith("SHOWER_ABSORB EX1_DEDUP "):
                        dedups.append(parse_kv(line))
                if not dedups:
                    continue
                dump = os.path.join(os.path.dirname(log), "calib-pr-evt%d.json" % ev)
                orphans = -1
                nseg_tot = -1
                if os.path.exists(dump):
                    j = json.load(open(dump))
                    segs = j.get("segments") or []
                    nseg_tot = len(segs)
                    orphans = sum(1 for s in segs if (s.get("shower_id") or -1) == -1)
                for d in dedups:
                    rows.append(dict(arm=os.path.basename(root), event=ev,
                                     orphans_final=orphans, nseg_event=nseg_tot, **d))

    n1 = [r for r in rows if r.get("old_nseg", 0) <= 1 and r.get("erase")]
    nm = [r for r in rows if r.get("old_nseg", 0) > 1 and r.get("erase")]
    nk = [r for r in rows if not r.get("erase")]
    print("EX1_DEDUP fires: %d  (erase single-seg wrapper: %d, ERASE MULTI-SEG: %d, kept: %d)"
          % (len(rows), len(n1), len(nm), len(nk)))
    for r in sorted(nm, key=lambda r: -r.get("old_nseg", 0)):
        print("  MULTI-SEG ERASE %(arm)s evt%(event)d into=%(into_start_seg)s "
              "old_shower=%(old_shower_id)s nseg=%(old_nseg)s kine=%(old_kine_mev)s MeV "
              "final_orphans=%(orphans_final)s/%(nseg_event)s" % r)
    if args.tsv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.tsv, "w") as fh:
            fh.write("\t".join(keys) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
        print("wrote %s (%d rows)" % (args.tsv, len(rows)))


if __name__ == "__main__":
    main()

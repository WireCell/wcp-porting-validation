#!/usr/bin/env python3
"""pr/130 item 4 part 10 -- why did the RIGHT shower not acquire the cluster?

Part 9 showed the shower walk never reaches the owner-approved charge (0/78
contention) and that 96.2% of it sits in clusters the shower does not hold, so
the acquisition has to happen at a cross-cluster absorber.  This joins the
WCT_SHOWER_XCLUS_DEBUG tape against those same 78 segments:

  OWNED     the target shower's candidate loop dropped the segment because
            ANOTHER shower already owned it -- decided before any geometry is
            computed, so no predicate at this site can ever see it.
  REJECTED  the target evaluated it and the pass-4 cone refused -- a threshold
            question, and the tape carries how far it missed by.
  ABSENT    neither -- the pair never entered the loop at all.

Repro:
  ./scripts/pr130_blocked_probe.sh xclus
  scripts/pr130_xclus_census.py > docs/pr/pr130-xclus-census.txt
"""
import collections
import csv
import glob
import os
import re
import sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_OWN = re.compile(r"SHOWER_XCLUS OWNED site=(\S+) shower=(-?\d+) seg=(-?\d+) owner=(-?\d+)")
RE_REJ = re.compile(r"SHOWER_XCLUS REJECT site=(\S+) shower=(-?\d+) seg=(-?\d+) "
                    r"angle_v1=([-\d.]+) angle_v2=([-\d.]+) pair_dis_cm=([-\d.]+)"
                    r"(?: tmp_dis_cm=([-\d.]+) close_dis_cm=([-\d.]+))?")


def main():
    miss = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(SX, "docs", "pr",
                                              "pr130-blocked-census.tsv")),
                            delimiter="\t"):
        miss[int(r["event"])][int(r["seg"])] = (int(r["shower"]), float(r["q"]))

    rows = []
    for ev, segs in sorted(miss.items()):
        own, rej = {}, {}
        for log in glob.glob(os.path.join(SX, "work-pr130r1-xcon-*",
                                          "pr_evt%d" % ev, "stdout.log")):
            for ln in open(log, errors="replace"):
                m = RE_OWN.search(ln)
                if m:
                    own[(int(m.group(2)), int(m.group(3)))] = (m.group(1), int(m.group(4)))
                    continue
                m = RE_REJ.search(ln)
                if m:
                    rej.setdefault((int(m.group(2)), int(m.group(3))), []).append(m.groups())
        for sid, (shw, q) in segs.items():
            o = own.get((shw, sid))
            r = rej.get((shw, sid))
            cls = "OWNED" if o else ("REJECTED" if r else "ABSENT")
            rows.append(dict(event=ev, shower=shw, seg=sid, q=q, cls=cls,
                             site=(o[0] if o else (r[0][0] if r else "")),
                             owner=(o[1] if o else -1),
                             angle_v1=(r[0][3] if r else ""),
                             angle_v2=(r[0][4] if r else ""),
                             pair_dis=(r[0][5] if r else "")))
    tot = sum(x["q"] for x in rows)
    print("=" * 78)
    print("pr/130 item 4 part 10 -- why the right shower did not acquire the cluster")
    print("=" * 78)
    print("\n%d segment(s), %.4e of charge\n" % (len(rows), tot))
    by = collections.Counter()
    qby = collections.Counter()
    for r in rows:
        by[r["cls"]] += 1
        qby[r["cls"]] += r["q"]
    for c in ("OWNED", "REJECTED", "ABSENT"):
        print("  %-9s %3d seg  %.4e  %5.1f%%" % (c, by[c], qby[c],
                                                 100 * qby[c] / tot if tot else 0))
    print("\nper event:")
    pe = collections.defaultdict(collections.Counter)
    for r in rows:
        pe[r["event"]][r["cls"]] += 1
    print("  %-8s %7s %9s %7s" % ("event", "OWNED", "REJECTED", "ABSENT"))
    for ev in sorted(pe):
        print("  %-8d %7d %9d %7d" % (ev, pe[ev]["OWNED"], pe[ev]["REJECTED"],
                                      pe[ev]["ABSENT"]))
    print("\nper segment (by charge):")
    print("  %-8s %-8s %-8s %-9s %10s %-18s %8s %8s %9s"
          % ("event", "shower", "seg", "class", "q", "site", "angle_v1",
             "angle_v2", "pair_dis"))
    for r in sorted(rows, key=lambda x: -x["q"]):
        print("  %-8d %-8d %-8d %-9s %10.3e %-18s %8s %8s %9s"
              % (r["event"], r["shower"], r["seg"], r["cls"], r["q"],
                 r["site"] or "-", r["angle_v1"] or "-", r["angle_v2"] or "-",
                 r["pair_dis"] or "-"))
    out = os.path.join(SX, "docs", "pr", "pr130-xclus-census.tsv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

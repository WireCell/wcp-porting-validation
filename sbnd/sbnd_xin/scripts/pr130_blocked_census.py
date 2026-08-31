#!/usr/bin/env python3
"""pr/130 item 4 part 8 -- did the RIGHT shower ever reach the segment it lost?

Joins the WCT_SHOWER_BLOCKED_DEBUG tape (emitted by PRShower.cxx's flood-fill
when a walk is turned away by used_segments) against the segments the owner's
2026-08-29 scan marked IN but the reconstruction does not hold.

  CONTENTION   the target shower's own walk emitted a BLOCKED line for that
               segment -> it reached the segment and was refused because
               something already held it.  Order-dependent; a reordering or
               revisit change can fix it.
  UNREACHED    no BLOCKED line from that shower -> its walk never got there at
               all, so processing order is irrelevant and the fix has to be a
               reach change.

Repro:
  cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
  ./scripts/pr130_blocked_probe.sh probe          # arms work-pr130r1-blkon-*
  scripts/pr130_blocked_census.py > docs/pr/pr130-blocked-census.txt
"""
import collections
import csv
import glob
import os
import re
import sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "em_display"))
import em117_score as S                                    # noqa: E402

RE_BLK = re.compile(r"SHOWER_ABSORB BLOCKED shower_start_seg=(-?\d+) seg=(-?\d+) "
                    r"pdg=(-?\d+) len_cm=([-\d.]+)")

# 463565 was ruled verbally, not through the label store: shower 13001 takes
# every segment of objects 115088 / 114086 / 109073 / 72064.
VERBAL = {463565: 13001}


def tape(ev):
    """-> {(walking shower, segment): (pdg, len_cm)}"""
    out = {}
    for log in glob.glob(os.path.join(SX, "work-pr130r1-blkon-*",
                                      "pr_evt%d" % ev, "stdout.log")):
        for ln in open(log, errors="replace"):
            m = RE_BLK.search(ln)
            if m:
                out[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)),
                                                           float(m.group(4)))
    return out


def main():
    man = S.load_manifest(os.path.join(SX, "em_display",
                                       "em130q-scan10-manifest.tsv"))
    labs = S.load_labels("emscan-0829-pr130qmiss")
    prepdir = os.path.join(SX, "em_display", "emprep-pr130scan10")

    rows = []
    for ev in sorted(set(list(labs) + list(VERBAL))):
        m = man.get(ev)
        if not m:
            continue
        dump = S.load_dump(m["dump"])
        prep = S.load_prep(ev, prepdir)
        if dump is None:
            continue
        actual, seginfo, _ = S.digest_dump(dump, prep)
        blk = tape(ev)
        rec = labs.get(ev)
        md = ((rec or {}).get("em") or {}).get("marks_detail") or {}
        if not md and ev in VERBAL:
            # verbal ruling: every member of the four named objects
            shw = VERBAL[ev]
            have = actual.get(shw, set())
            want = set()
            for obj in (115088, 114086, 109073, 72064):
                want |= actual.get(obj, set())
            md = None
            pairs = [(shw, sorted(want - have))]
        else:
            pairs = []
            for shw, det in md.items():
                r = S.score_shower(int(shw), det, actual, seginfo, cross_run=True)
                if r and r["miss"]:
                    pairs.append((r["matched"] if r["matched"] > 0 else int(shw),
                                  list(r["miss"])))
        for shw, miss in pairs:
            for sid in miss:
                hit = blk.get((shw, sid))
                rows.append(dict(event=ev, shower=shw, seg=sid,
                                 cls="CONTENTION" if hit else "UNREACHED",
                                 q=seginfo.get(sid, {}).get("charge", 0.0),
                                 length=seginfo.get(sid, {}).get("length", 0.0),
                                 pdg=seginfo.get(sid, {}).get("pdg")))
    print("=" * 78)
    print("pr/130 item 4 part 8 -- contention vs reach, on the owner-approved merges")
    print("=" * 78)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    tot = sum(r["q"] for r in rows)
    print("\n%d segment(s) the scan wants and the reco does not hold, %.4e of charge\n"
          % (len(rows), tot))
    for cls in ("CONTENTION", "UNREACHED"):
        rr = by.get(cls, [])
        q = sum(x["q"] for x in rr)
        print("  %-12s %3d seg  %.4e  %5.1f%%"
              % (cls, len(rr), q, 100 * q / tot if tot else 0))
    print("\nper event:")
    pe = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
    for r in rows:
        e = pe[r["event"]]
        i = 0 if r["cls"] == "CONTENTION" else 1
        e[i] += 1
        e[2 + i] += r["q"]
    print("  %-8s %10s %10s %12s %12s" % ("event", "contend", "unreached",
                                          "q_contend", "q_unreach"))
    for ev in sorted(pe):
        a, b, qa, qb = pe[ev]
        print("  %-8d %10d %10d %12.3e %12.3e" % (ev, a, b, qa, qb))
    print("\nper segment:")
    print("  %-8s %-8s %-8s %-12s %10s %6s %5s"
          % ("event", "shower", "seg", "class", "q", "len", "pdg"))
    for r in sorted(rows, key=lambda x: -x["q"]):
        print("  %-8d %-8d %-8d %-12s %10.3e %6.1f %5s"
              % (r["event"], r["shower"], r["seg"], r["cls"], r["q"],
                 r["length"], r["pdg"]))
    out = os.path.join(SX, "docs", "pr", "pr130-blocked-census.tsv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

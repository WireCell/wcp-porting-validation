#!/usr/bin/env python3
"""doc sbnd_xin/119 round 5: four allocators, one invocation, one instrument at a time.

Round 4 sec 10.4 compared a jemalloc number to a tcmalloc number that came from a DIFFERENT
INSTRUMENT and a DIFFERENT INVOCATION -- the jemalloc 1.343 GiB was the in-job MEM ladder under
scripts/perf/profile_pr119r4.sh with heap sampling active; the tcmalloc 2.084 GiB was .time.meta
maxrss_kb from a full run_pr_chain_batch.sh run.  On these seven events getrusage reads 0.15-0.19
GiB above the ladder, and read ladder-against-ladder the p50 comparison reverses sign.  This
script prints every arm under every instrument so that mistake cannot be repeated by reading it.

  maxrss  GiB   getrusage(RUSAGE_CHILDREN) high-water -- a kernel number, the one an operational
                memory cap is actually set against.  THE CLAIM INSTRUMENT for this round.
  ladder  GiB   the in-job MEM ladder's peak resident -- VmRSS sampled at stage boundaries, so it
                misses a peak that lives inside a stage.  Shown only to keep round 4 readable.
  TICK       s  the clustering node's own compute.  Survives batch contention; the arms are run
                back to back at one fan-out so this is comparable.
  wall       s  context only.

Usage:  python3 scripts/d119/r5_alloc.py [tail|cv|nuecc|off ...] > docs/119_figs/119_r5_alloc.txt
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_rank import scan                                                        # noqa: E402

SX = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

# Arm-name templates per sample.  The tail arms carry a `t` so they cannot collide with the
# 62-event arms of the same allocator (both live under work-r3nue-*).
#
# THE 62-EVENT ARMS MUST CARRY `r5`, and the reason is a trap this script walked into once:
# work-r3cv-d119tcm ALREADY EXISTS and is ROUND 1's arm, produced on the PRE-ROUND-3 binary.
# Reading round 5's jemalloc against it silently folds round 3's -14.9 % CPU win into what would
# be reported as an allocator delta.  Round 5's own arms are work-r3<s>-d119r5<tag>; `jem` is the
# one exception and stays bare because lever_gate.py addresses an arm by its LEVER name.
ARMS = {"tail": "work-r3nue-d119t%s",
        "cv": "work-r3cv-d119%s", "nuecc": "work-r3nue-d119%s", "off": "work-r3off-d119%s"}
GTAG = {"tcm": "r5tcm", "glb": "r5glb", "jem": "jem", "rel": "r5rel"}    # 62-event arm suffixes
ALLOC = [("tcm", "PRODUCTION: libtcmalloc_minimal joined to the preload (round 1)"),
         ("glb", "glibc malloc -- the pre-round-1 environment, and doc 116's control"),
         ("jem", "jemalloc 5.3.0, no MALLOC_CONF (round 4's runs had the heap sampler ON)"),
         ("rel", "PRODUCTION tcmalloc + TCMALLOC_RELEASE_RATE=10 -- the MECHANISM control")]
COLS = ((4, "maxrss  GiB"), (2, "ladder  GiB"), (1, "TICK      s"), (3, "wall      s"))


def fmt(v):
    return "  n/a " if v is None else "%6.3f" % v


def main():
    samples = sys.argv[1:] or ["tail", "nuecc", "cv", "off"]
    print("# doc sbnd_xin/119 round 5 -- allocator arms, back to back on one box, one fan-out.")
    print("# CLAIM INSTRUMENT = maxrss (getrusage CHILDREN high-water).  Every column below is")
    print("# read from the SAME instrument across arms; round 4's error was mixing two.")
    for s in samples:
        got = {}
        for tag, _ in ALLOC:
            arm = ARMS[s] % (tag if s == "tail" else GTAG[tag])
            # Refuse to read an arm this round did not produce.  The marker is the same one
            # stageB_alloc.sh's M13 guard writes; without this check a reader can quietly pick up
            # a foreign arm that a writer would have been refused (see the ARMS note above).
            if not os.path.isdir(os.path.join(SX, arm)):
                continue
            mark = os.path.join(SX, arm, ".d119_arm")
            got_mark = open(mark).read().strip() if os.path.exists(mark) else None
            if not (got_mark or "").startswith("r5-"):
                sys.exit("REFUSING: %s carries marker %r, not a round-5 arm." % (arm, got_mark))
            got[tag] = scan(arm)
        if not got:
            continue
        base = "tcm" if "tcm" in got else sorted(got)[0]
        common = sorted(set.intersection(*(set(v) for v in got.values())),
                        key=lambda k: (k[0], int(k[1])))
        print("\n" + "=" * 92)
        print("## sample %s -- %d events paired across %d arms (%s)"
              % (s, len(common), len(got), ", ".join(sorted(got))))
        print("=" * 92)
        if not common:
            print("   NO PAIRED EVENTS")
            continue

        for tag, label in ALLOC:
            if tag in got:
                print("   %-4s %s" % (tag, label))

        # Per-event maxrss, because the whole point of the tail set is that the mean hides it.
        if s == "tail":
            print("\n   peak RSS (getrusage), GiB, per event")
            print("   %-16s %s" % ("event", "".join("%9s" % t for t, _ in ALLOC if t in got)))
            for k in common:
                print("   %-16s %s" % ("/".join(k),
                                       "".join("%9s" % fmt(got[t][k][4]) for t, _ in ALLOC
                                               if t in got)))

        for idx, name in COLS:
            vals = {t: [got[t][k][idx] for k in common if got[t][k][idx] is not None]
                    for t in got}
            if not all(vals.values()):
                continue
            print("\n   %s" % name)
            b = st.mean(vals[base])
            bmax = max(vals[base])
            for tag, _ in ALLOC:
                if tag not in got:
                    continue
                m, mx = st.mean(vals[tag]), max(vals[tag])
                print("      %-4s mean %8.3f (%+6.1f %%)   max %8.3f (%+6.1f %%)"
                      % (tag, m, 100.0 * (m - b) / b if b else 0.0,
                         mx, 100.0 * (mx - bmax) / bmax if bmax else 0.0))
            print("      (%% against %s, the production allocator)" % base)

    print("\n" + "=" * 92)
    print("HOW TO READ A DELTA HERE.  Round 3's null pair measured the run-to-run floor on this")
    print("machine at +0.2 / -1.6 / -2.6 % on TICK (doc 119 sec 9.3).  A TICK delta inside that")
    print("band is not a measurement.  maxrss is far more repeatable than TICK, but it is still a")
    print("single run per event: a 1-2 % memory delta is not a finding either.")


if __name__ == "__main__":
    main()

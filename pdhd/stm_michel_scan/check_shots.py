#!/usr/bin/env python3
"""doc pdvd/55 sec 13.2, doc pdvd/69 -- find 3-D frames a lost WebGL context spoiled.

    ./check_shots.py SHOTSDIR [RESHOOT_OUT.txt] [--min-colours 1000]

SHOTSDIR is a `scan_harness.py shots --out` directory.  RESHOOT_OUT, if given, is
written one key per line -- the format `shots --items-file` reads -- and is empty
when nothing needs re-shooting.  Exit 1 if any item is named, else 0.

Several headless chromium instances at once exceed the WebGL context budget
(doc 55 sec 13.2: five do, two do not).  The losing process's 3-D panel then
renders one of two ways, and neither looks broken to a scanner:

  BLANK    the empty canvas.  Every blank frame is the SAME canvas, so it is
           byte-identical to every other one: a 3-D frame whose md5 is shared by
           three or more distinct items is blank.  A sparse-but-real frame is
           unique.  File size does not work -- a faint real track compresses to
           ~7 kB too (039252_15/81).
  SMEARED  solid wedges and filled planes where the point cloud should be.
           Unique per item and normal in size, so no md5 or size test sees it.
           It carries few distinct colours: c_3d_stop.png below 1000 unique
           colours.  Measured on the 509 PDVD tranche-2 items: the three clean
           processes gave min 1168 (median 3877), 0 of 305 below 1000; the two
           context-lost processes gave median 722, 187 of 204 below 1000.  The
           threshold is that PDVD measurement at the harness's fixed 45 cm zoom
           and 760x760 panel; it has not been measured on PDHD.

The third test is the harness's own record.  `scan_harness.py shots` writes
SHOTSDIR/_webgl_lost.txt naming every item shot at or after the first sign of a
context loss in that process (doc 69).  That is the process rule of doc 55 sec
13.2 -- "if a shots process lost its context, what it drew after that is
suspect" -- at item resolution, and it is the test to act on: the colour count
missed 17 of 204 lost-process items.  A shots dir written before the harness
recorded this has no such file, and the line says so.

Committed from the tranche-2 scratch copy doc 55's repro was calling
(never committed); the one change is that the process rule now reads the
harness's record instead of a log-to-chunk map in that scratch dir.
"""
import argparse, collections, hashlib, os, sys

F3 = ["b_3d_wide.png", "c_3d_stop.png", "d_3d_stop.png", "e_3d_stop.png"]
ALL = ["a_proj_full.png", "f_meas.png", "g_dqdx.png"] + F3
LOST_FILE = "_webgl_lost.txt"          # written by scan_harness.py do_shots


def key_of(d):
    """039252_12_114 -> 039252_12/114 (scan_harness writes key.replace('/', '_'))."""
    return "%s_%s/%s" % tuple(d.rsplit("_", 2))


def unique_colours(fp):
    from PIL import Image
    im = Image.open(fp).convert("RGB")
    return len(im.getcolors(maxcolors=1 << 24) or [])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shotsdir")
    ap.add_argument("reshoot_out", nargs="?")
    ap.add_argument("--min-colours", type=int, default=1000)
    a = ap.parse_args(argv)

    dirs = sorted(d for d in os.listdir(a.shotsdir)
                  if os.path.isdir(os.path.join(a.shotsdir, d)))
    if not dirs:                         # a wrong path must not print "clean"
        print("REFUSE: no item dirs under %s" % a.shotsdir, file=sys.stderr)
        return 2

    by_md5 = collections.defaultdict(set)
    incomplete, colours = [], {}
    for d in dirs:
        p = os.path.join(a.shotsdir, d)
        miss = [f for f in ALL + ["context.json"] if not os.path.exists(os.path.join(p, f))]
        if miss:
            incomplete.append((d, miss))
        for f in F3:
            fp = os.path.join(p, f)
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    by_md5[hashlib.md5(fh.read()).hexdigest()].add((d, f))
        fp = os.path.join(p, "c_3d_stop.png")
        if os.path.exists(fp):
            colours[d] = unique_colours(fp)

    blank = collections.defaultdict(list)
    for v in by_md5.values():
        if len({d for d, _ in v}) > 2:
            for d, f in v:
                blank[d].append(f)
    smeared = sorted(d for d, n in colours.items() if n < a.min_colours)

    lost_fp = os.path.join(a.shotsdir, LOST_FILE)
    lost = None
    if os.path.exists(lost_fp):
        lost = [l.strip() for l in open(lost_fp) if l.strip() and not l.startswith("#")]

    print("%d item dirs, %d distinct 3-D frames" % (len(dirs), len(by_md5)))
    print("incomplete dirs: %d" % len(incomplete))
    for d, m in incomplete[:10]:
        print("   %s missing %s" % (d, m))
    print("blank 3-D frames (md5 shared by 3+ items): %d frames over %d items"
          % (sum(len(v) for v in blank.values()), len(blank)))
    for d in sorted(blank):
        print("   %-18s %s" % (d, " ".join(sorted(blank[d]))))
    cs = sorted(colours.values())
    print("c_3d_stop.png unique colours: n %d, min %s, median %s; below %d: %d"
          % (len(cs), cs[0] if cs else "-", cs[len(cs) // 2] if cs else "-",
             a.min_colours, len(smeared)))
    for d in smeared[:10]:
        print("   %-18s %d" % (d, colours[d]))
    if lost is None:
        print("harness context-loss record: none (%s absent -- written before doc 69, "
              "or by an older harness)" % LOST_FILE)
    else:
        print("harness context-loss record: %d item(s) shot at/after a context loss" % len(lost))
        for k in lost[:10]:
            print("   %s" % k)

    keys = {key_of(d) for d in blank} | {key_of(d) for d, _ in incomplete} \
        | {key_of(d) for d in smeared} | set(lost or [])
    keys = sorted(keys)
    if a.reshoot_out:
        with open(a.reshoot_out, "w") as fh:
            fh.write("".join(k + "\n" for k in keys))
        print("\nwrote %s with %d item(s) to re-shoot" % (a.reshoot_out, len(keys)))
    print("VERDICT: %s" % ("%d item(s) to re-shoot" % len(keys) if keys else "clean"))
    return 1 if keys else 0


if __name__ == "__main__":
    sys.exit(main())

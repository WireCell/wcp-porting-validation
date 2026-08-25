#!/usr/bin/env python3
"""doc 82 -- assemble a stage-A GROUP directory out of per-event products.

The exact inverse of split_group_products.py, and it exists for one reason: the
doc 81 sec 7.1 Q/L non-determinism reproducer.  Doc 81 built its 2-event group
by hand and never saved it, and round 5 then pruned the group scratch, so the
group npz/opflash that showed the bug no longer exist anywhere.  What DOES
survive is the per-event split of them, which is lossless -- so a group can be
rebuilt from any subset of events without re-running the reco1 dump or imaging.

  <src>/evt<ID>/icluster-apa{0,1}-{active,masked}.npz  members cluster_<EVT>_*
  <src>/ql_evt<ID>/opflash_apa{0,1}.tar.gz             members opflash_tensorset_<EVT>_*

Members are emitted in the order the events are given on the command line, and
within an event in the order the per-event archive holds them -- i.e. exactly
the layout the group job wrote before the split.  Member ORDER is load-bearing
for a group archive (doc 76 round 2), so it is not sorted or regrouped here.

usage: merge_group_products.py <src_root> <group_dir> <evt> [<evt> ...]
"""
import os, sys, tarfile, zipfile

NPZ = ["icluster-apa0-active", "icluster-apa0-masked",
       "icluster-apa1-active", "icluster-apa1-masked"]


def merge_npz(src, gdir, evts):
    for base in NPZ:
        srcs = [os.path.join(src, "evt" + e, base + ".npz") for e in evts]
        # ql_evt<ID>/ holds symlinks to ../evt<ID>/; accept either spelling.
        srcs = [p if os.path.exists(p)
                else os.path.join(src, "ql_evt" + os.path.basename(os.path.dirname(p))[3:],
                                  base + ".npz")
                for p in srcs]
        missing = [p for p in srcs if not os.path.exists(p)]
        if missing:
            raise SystemExit("missing imaging npz: %s" % missing[0])
        n = 0
        # ZIP_STORED: an npz is uncompressed and hash_archive.py compares the
        # member payloads, not the container (CLAUDE.md M2).
        with zipfile.ZipFile(os.path.join(gdir, base + ".npz"), "w",
                             zipfile.ZIP_STORED) as o:
            for p in srcs:
                z = zipfile.ZipFile(p)
                for name in z.namelist():      # per-event order preserved
                    o.writestr(name, z.read(name))
                    n += 1
        print("  %s: %d members from %d events" % (base, n, len(evts)))


def merge_opflash(src, gdir, evts):
    for apa in (0, 1):
        srcs = [os.path.join(src, "ql_evt" + e, "opflash_apa%d.tar.gz" % apa)
                for e in evts]
        missing = [p for p in srcs if not os.path.exists(p)]
        if missing:
            raise SystemExit("missing opflash: %s" % missing[0])
        n = 0
        with tarfile.open(os.path.join(gdir, "opflash_apa%d.tar.gz" % apa), "w:gz") as o:
            for p in srcs:
                t = tarfile.open(p)
                for ti in t:
                    if ti.isfile():
                        o.addfile(ti, t.extractfile(ti))
                        n += 1
        print("  opflash_apa%d: %d members from %d events" % (apa, n, len(evts)))


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    src, gdir, evts = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(gdir, exist_ok=True)
    print("assembling %s from %s (%d events: %s)"
          % (gdir, src, len(evts), " ".join(evts)))
    merge_npz(src, gdir, evts)
    merge_opflash(src, gdir, evts)
    with open(os.path.join(gdir, "events.txt"), "w") as f:
        f.write("\n".join(evts) + "\n")
    print("ok")


if __name__ == "__main__":
    main()

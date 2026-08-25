#!/usr/bin/env python3
"""doc 81 round 1 -- split a stage-A GROUP directory's shared products into the
per-event layout a per-event job writes.

The Q/L job writes its own per-event products directly (evt_subdir), but two
stage-A products are shared by the whole group and have no %-template:

  icluster-apa{0,1}-{active,masked}.npz   members named cluster_<EVT>_<kind>.npy
  opflash_apa{0,1}.tar.gz                 members named opflash_tensorset_<EVT>_*

Both are keyed by event id, so the split is exact: every member goes to exactly
one event and no member is dropped (the script verifies this and fails if not).

Layout produced, matching run_img_evt.sh / run_ql_evt.sh:
  <root>/evt<ID>/icluster-apa{0,1}-{active,masked}.npz
  <root>/ql_evt<ID>/opflash_apa{0,1}.tar.gz
  <root>/ql_evt<ID>/icluster-*.npz -> ../evt<ID>/icluster-*.npz   (symlink, as legacy)

usage: split_group_products.py <group_dir> <out_root>
"""
import os, re, sys, tarfile, zipfile

NPZ = ["icluster-apa0-active", "icluster-apa0-masked",
       "icluster-apa1-active", "icluster-apa1-masked"]
RE_CLUSTER = re.compile(r"^cluster_(\d+)_")
RE_OPFLASH = re.compile(r"^opflash_tensor(?:set)?_(\d+)[_.]")


def split_npz(gdir, out, evts):
    for base in NPZ:
        src = os.path.join(gdir, base + ".npz")
        if not os.path.exists(src):
            continue
        z = zipfile.ZipFile(src)
        per = {}
        for n in z.namelist():
            m = RE_CLUSTER.match(n)
            if not m:
                raise SystemExit("unkeyed member %r in %s" % (n, src))
            per.setdefault(m.group(1), []).append(n)
        unknown = set(per) - set(evts)
        if unknown:
            raise SystemExit("members for events not in the group: %s" % sorted(unknown))
        written = 0
        for e, names in per.items():
            d = os.path.join(out, "evt" + e)
            os.makedirs(d, exist_ok=True)
            # ZIP_STORED: npz is uncompressed, and the member payloads are what
            # hash_archive.py compares (CLAUDE.md M2).
            with zipfile.ZipFile(os.path.join(d, base + ".npz"), "w", zipfile.ZIP_STORED) as o:
                for n in names:                       # source order preserved
                    o.writestr(n, z.read(n))
                    written += 1
        if written != len(z.namelist()):
            raise SystemExit("%s: split %d of %d members" % (src, written, len(z.namelist())))
        print("  %s: %d members -> %d events" % (base, written, len(per)))


def split_opflash(gdir, out, evts):
    for apa in (0, 1):
        src = os.path.join(gdir, "opflash_apa%d.tar.gz" % apa)
        if not os.path.exists(src):
            continue
        t = tarfile.open(src)
        per = {}
        for ti in t:
            if not ti.isfile():
                continue
            m = RE_OPFLASH.match(os.path.basename(ti.name))
            if not m:
                raise SystemExit("unkeyed member %r in %s" % (ti.name, src))
            per.setdefault(m.group(1), []).append(ti)
        written = 0
        for e, tis in per.items():
            d = os.path.join(out, "ql_evt" + e)
            os.makedirs(d, exist_ok=True)
            with tarfile.open(os.path.join(d, "opflash_apa%d.tar.gz" % apa), "w:gz") as o:
                for ti in tis:
                    o.addfile(ti, t.extractfile(ti))
                    written += 1
        print("  opflash_apa%d: %d members -> %d events" % (apa, written, len(per)))


def link_imaging(out, evts):
    for e in evts:
        qd = os.path.join(out, "ql_evt" + e)
        os.makedirs(qd, exist_ok=True)
        for base in NPZ:
            tgt = os.path.join("..", "evt" + e, base + ".npz")
            lnk = os.path.join(qd, base + ".npz")
            if os.path.lexists(lnk):
                os.unlink(lnk)
            if os.path.exists(os.path.join(out, "evt" + e, base + ".npz")):
                os.symlink(tgt, lnk)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    gdir, out = sys.argv[1], sys.argv[2]
    evts = open(os.path.join(gdir, "events.txt")).read().split()
    if not evts:
        raise SystemExit("no events.txt in " + gdir)
    print("splitting %s -> %s (%d events)" % (gdir, out, len(evts)))
    split_npz(gdir, out, evts)
    split_opflash(gdir, out, evts)
    link_imaging(out, evts)
    print("ok")


if __name__ == "__main__":
    main()

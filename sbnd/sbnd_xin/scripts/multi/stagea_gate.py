#!/usr/bin/env python3
"""doc 81 -- gate a stage-A work root against the recorded per-event products.

Nothing in this tree compared imaging or Q/L products between two work roots
before this (scripts/multi/ql_legacy_gate.sh re-RUNS Q/L per event and symlinks
imaging in, so it gates a binary change, not a layout or a group).

Compares MEMBER CONTENT, never raw archive bytes -- tar/zip embed mtimes, so a
byte compare of the container is meaningless (CLAUDE.md M2).

usage: stagea_gate.py <new_root> --img <work-img-<s>> --ql <work-<s>-ql0819>
                      [--events E [E ...]] [--jobs N]
"""
import argparse, hashlib, os, sys, tarfile, zipfile
from concurrent.futures import ProcessPoolExecutor

NPZ = ["icluster-apa0-active", "icluster-apa0-masked",
       "icluster-apa1-active", "icluster-apa1-masked"]
QL = ["mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip"]


def members(path):
    """name -> sha256(payload) for a zip or a tar."""
    if not os.path.exists(path):
        return None
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    t = tarfile.open(path)
    return {ti.name: hashlib.sha256(t.extractfile(ti).read()).hexdigest()
            for ti in t if ti.isfile()}


def cmp_one(args):
    evt, new_root, img, ql = args
    rows = []
    for base in NPZ:
        rows.append(("img/" + base,
                     os.path.join(img, "evt%s" % evt, base + ".npz"),
                     os.path.join(new_root, "evt%s" % evt, base + ".npz")))
    for f in QL + ["pctree-evt%s.tar.gz" % evt]:
        rows.append(("ql/" + f,
                     os.path.join(ql, "ql_evt%s" % evt, f),
                     os.path.join(new_root, "ql_evt%s" % evt, f)))
    out = []
    for label, a, b in rows:
        ma, mb = members(a), members(b)
        if ma is None or mb is None:
            out.append((evt, label, "MISSING", "ref" if ma is None else "new"))
            continue
        if ma == mb:
            out.append((evt, label, "SAME", len(ma)))
        else:
            d = sum(1 for k in ma if mb.get(k) != ma[k]) + sum(1 for k in mb if k not in ma)
            out.append((evt, label, "DIFFER", d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_root")
    ap.add_argument("--img", required=True)
    ap.add_argument("--ql", required=True)
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    evts = a.events or sorted(
        (d[len("ql_evt"):] for d in os.listdir(a.new_root) if d.startswith("ql_evt")),
        key=int)
    if not evts:
        sys.exit("no ql_evt* under " + a.new_root)

    work = [(e, a.new_root, a.img, a.ql) for e in evts]
    n_same = n_diff = n_miss = 0
    bad = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for rows in ex.map(cmp_one, work):
            for evt, label, verdict, extra in rows:
                if verdict == "SAME":
                    n_same += 1
                elif verdict == "DIFFER":
                    n_diff += 1; bad.append((evt, label, "%s members differ" % extra))
                else:
                    n_miss += 1; bad.append((evt, label, "missing on the %s side" % extra))

    print("# events: %d  archives compared: %d" % (len(evts), n_same + n_diff + n_miss))
    for evt, label, why in bad[:40]:
        print("  evt %-9s %-28s %s" % (evt, label, why))
    if len(bad) > 40:
        print("  ... and %d more" % (len(bad) - 40))
    if n_diff or n_miss:
        print("FAIL  same=%d differ=%d missing=%d" % (n_same, n_diff, n_miss))
        return 1
    print("PASS all %d archives member-content identical" % n_same)
    return 0


if __name__ == "__main__":
    sys.exit(main())

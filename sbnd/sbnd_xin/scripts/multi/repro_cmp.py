#!/usr/bin/env python3
"""doc 82 -- compare the Q/L products of two per-event-layout roots.

stagea_gate.py gates a whole work root against the recorded img+ql arms and
takes its event list from the root itself.  The reproducer needs something
smaller and symmetric: two arbitrary roots, an explicit event list, Q/L
products only (imaging is the shared input here, not an output), and a one-line
verdict naming the first differing member so a draw-vs-draw matrix stays
readable.

Member CONTENT only -- tar/zip embed mtimes (CLAUDE.md M2).

usage: repro_cmp.py <root_a> <root_b> <evt> [<evt> ...]
"""
import hashlib, os, sys, tarfile, zipfile

QL = ["mabc-all-apa.zip", "mabc-apa0-face0.zip", "mabc-apa1-face0.zip",
      "pctree-evt%s.tar.gz"]


def members(path):
    if not os.path.exists(path):
        return None
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    t = tarfile.open(path)
    return {ti.name: hashlib.sha256(t.extractfile(ti).read()).hexdigest()
            for ti in t if ti.isfile()}


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    a, b, evts = sys.argv[1], sys.argv[2], sys.argv[3:]
    same = diff = miss = 0
    first = ""
    for e in evts:
        for base in QL:
            name = base % e if "%s" in base else base
            pa = os.path.join(a, "ql_evt" + e, name)
            pb = os.path.join(b, "ql_evt" + e, name)
            ma, mb = members(pa), members(pb)
            if ma is None or mb is None:
                miss += 1
                if not first:
                    first = "missing %s/%s" % ("A" if ma is None else "B", name)
                continue
            if ma == mb:
                same += 1
                continue
            diff += 1
            if not first:
                bad = sorted(k for k in set(ma) | set(mb)
                             if ma.get(k) != mb.get(k))
                first = "evt %s %s: %d/%d members differ, first %s" % (
                    e, name, len(bad), len(set(ma) | set(mb)), bad[0])
    verdict = "IDENTICAL" if diff == 0 and miss == 0 else "DIFFER"
    print("%s  same=%d differ=%d missing=%d%s"
          % (verdict, same, diff, miss, "   [%s]" % first if first else ""))
    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())

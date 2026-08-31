#!/usr/bin/env python3
"""doc 81 -- freeze the member-content hashes of a PR arm before it is retired.

The 08-25 removal set includes every ``work-pr112i-snapD2-<s>`` arm, which is
the only PER-EVENT reference at the current operating point (``prod0823`` is
pre-flip, see doc 81 sec 2).  Retiring an arm after its gate is the normal
pattern here, but that pattern relies on the gate staying re-checkable, and a
gate whose reference has been deleted is not.  This writes the reference side
of the doc-81 sec 8.1 gate to a small text file so the PASS can be re-verified
against ``work-<s>-prod0825`` for as long as that arm exists.

Covers BOTH halves of the sec 8.1 gate, because neither tool subsumes the
other and pr85 never opens the ROOT file:

* archives -- exactly the rollup ``pr85_hash_gate.py`` compares: sha256 over
  (member_name + payload) per member, members sorted by name.
* ``tracking-pr.root`` -- a sha256 over the canonical JSON of
  ``pr94_root_gate.py``'s own ``load()``, i.e. every tree, every branch, every
  entry, materialised to plain Python.  NOT a sha256 of the file: ROOT embeds
  write timestamps in its keys, so a raw file hash would report a difference
  that isn't one (M2).  NaN is normalised the way pr94 normalises it, so NaN
  positions still have to match.

A later check is a plain ``diff`` against a re-run of this script on the
surviving arm.

Usage: hash_manifest_20260825.py <arm> [<arm> ...]   (writes state-20260825/hashes/<arm>.tsv)
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pr85_hash_gate import rollup, archives_of  # noqa: E402
from pr94_root_gate import load as root_load  # noqa: E402

import re

EVT_RE = re.compile(r"pr_evt(\d+)$")

OUT = os.path.join(HERE, "state-20260825", "hashes")


def one(job):
    evt, path = job
    if path.endswith(".root"):
        trees = root_load(path)
        blob = json.dumps(trees, sort_keys=True, separators=(",", ":"))
        n = sum(len(v) for v in trees.values() if isinstance(v, dict))
        return evt, os.path.basename(path), hashlib.sha256(blob.encode()).hexdigest(), n
    h, n = rollup(path)
    return evt, os.path.basename(path), h, n


def main():
    arms = sys.argv[1:]
    if not arms:
        sys.exit("usage: hash_manifest_20260825.py <arm> [<arm> ...]")
    os.makedirs(OUT, exist_ok=True)
    for arm in arms:
        arm = arm.rstrip("/")
        jobs = []
        for d in sorted(os.listdir(arm)):
            m = EVT_RE.match(d)
            if not m:
                continue
            evt = int(m.group(1))
            pdir = os.path.join(arm, d)
            for p in archives_of(pdir, evt):
                jobs.append((evt, p))
            rp = os.path.join(pdir, "tracking-pr.root")
            if os.path.exists(rp):
                jobs.append((evt, rp))
        rows = []
        with ProcessPoolExecutor(max_workers=10) as ex:
            for evt, name, h, n in ex.map(one, jobs, chunksize=8):
                rows.append((evt, name, h, n))
        rows.sort()
        dest = os.path.join(OUT, os.path.basename(arm) + ".tsv")
        with open(dest, "w") as fp:
            fp.write("# doc 81 sec 8.1 reference-side member-content hashes\n")
            fp.write("# arm=%s products=%d\n" % (arm, len(rows)))
            fp.write("#evt\tproduct\trollup_sha256\tmembers_or_branches\n")
            for evt, name, h, n in rows:
                fp.write("%d\t%s\t%s\t%d\n" % (evt, name, h, n))
        print("%-38s %5d products -> %s" % (arm, len(rows), dest))


if __name__ == "__main__":
    main()

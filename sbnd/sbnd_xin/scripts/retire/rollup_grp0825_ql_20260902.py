#!/usr/bin/env python3
"""Freeze the record of work-*-grp0825/ql_evt* before the 2026-09-02 round
deletes it in place.

WHY THIS EXISTS.  grp0825 is a PROTECTED arm (PROTECTED.txt, doc 81: "the
input the MCS and PR rounds consume") but the protection is really about its
IMAGING half: 10478 symlinks across the tree resolve into grp0825/evt<ID>.
Nothing anywhere symlinks to grp0825/ql_evt<ID>, and that Q/L is two operating
points stale (doc 97 sec 2 measured today's knob-off run NOT reproducing it --
epoch drift, not nondeterminism).  So the ql half is retired and the imaging
half stays.

That is a PARTIAL deletion of a protected arm, which M13 does not let you do
on a hunch.  This writes the member-level rollup first -- same role
state-20260825/hashes/*.tsv played when doc 81 retired arms whose only
reference was a frozen hash side.  After this, the claim "grp0825's Q/L was
X" stays checkable even though the bytes are gone.

Output: state-20260902/grp0825-ql-rollup.tsv  (arm, event, file, size, sha256)
"""
import hashlib, os, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
OUT = os.path.join(ROOT, "scripts", "retire", "state-20260902", "grp0825-ql-rollup.tsv")
os.chdir(ROOT)

def digest(a):
    arm, d = a
    rows = []
    p = os.path.join(arm, d)
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        if not os.path.isfile(fp) or os.path.islink(fp):
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        rows.append((arm, d, f, os.path.getsize(fp), h.hexdigest()))
    return rows

jobs = []
for s in ("mcp1k", "mcp2k", "ncpi0", "nuecc48"):
    arm = f"work-{s}-grp0825"
    jobs += [(arm, d) for d in sorted(os.listdir(arm)) if d.startswith("ql_evt")]
print(f"{len(jobs)} ql_evt dirs to roll up", flush=True)

n = 0
with open(OUT, "w") as out, ProcessPoolExecutor(max_workers=8) as ex:
    out.write("arm\tevent\tfile\tsize\tsha256\n")
    for rows in ex.map(digest, jobs, chunksize=16):
        for r in rows:
            out.write("\t".join(map(str, r)) + "\n"); n += 1
print(f"wrote {n} member rows -> {OUT}", flush=True)

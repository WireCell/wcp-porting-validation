#!/usr/bin/env python3
"""Archive of the record layer of the doc pr/97 crash-sweep retirement,
2026-08-20 -- the gojsonnet-crash investigation's 484 work-pr97* dirs.

The investigation is closed (97_address_dependent_pr_chain.md: "TWO DEFECTS
FOUND, BOTH FIXED, GATED"). Every load-bearing number -- the crash-rate
table, both gdb backtraces, the byte-identical gate PASSes -- is already
quoted verbatim in the doc. This script drops the reproducible heavy blobs
(mabc zips, pctree/opflash/tracking archives) exactly as every prior
retirement round has, and archives only the small record layer (logs,
.status, compiled config, scripts) as a integrity-checked tar.gz per arm.

Fork of archive_records_20260819b.py's HEAVY classification, simplified:
no PROTECTED/KEEP/group logic needed -- every work-pr97* dir here is
scoped to one already-closed, self-contained investigation, named nowhere
else (checked: absent from PROTECTED.txt, docs/work-tags.md, and every
other docs/pr/*.md).

Output tree archive/records/pr97-crash-sweep-20260820/sweep/<tag>.tar.gz
(+ .links.txt + .manifest.tsv). Reads only; never touches work-pr97* itself.
"""
import os, re, tarfile, collections, sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
OUT = os.path.join(ROOT, "archive", "records", "pr97-crash-sweep-20260820", "sweep")
os.chdir(ROOT)

HEAVY = [("pctree",   re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",     re.compile(r'^mabc.*\.zip$')),
         ("calib",    re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$')),
         ("npz",      re.compile(r'.*\.npz$')),
         ("clusters", re.compile(r'^clusters-apa.*\.tar\.gz$')),
         ("opflash",  re.compile(r'^opflash_apa.*\.tar\.gz$')),
         ("tracking", re.compile(r'^tracking-pr\.root$')),
         ("oc56scan", re.compile(r'^oc56scan-evt.*\.jsonl$')),
         ("core",     re.compile(r'^core(\..*)?$'))]


def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f):
            return name
    return None


def archive_one(tag):
    os.makedirs(OUT, exist_ok=True)
    tgz = os.path.join(OUT, tag + ".tar.gz")
    keep, links = [], []
    cls = collections.defaultdict(lambda: [0, 0])
    for cur, sub, files in os.walk(tag):
        for name in list(sub):
            p = os.path.join(cur, name)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}")
                sub.remove(name)
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}")
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            hc = heavy_class(f)
            cls[hc or "record"][0] += 1
            cls[hc or "record"][1] += sz
            if hc is None:
                keep.append(p)
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(keep):
            tf.add(p)
    with open(os.path.join(OUT, tag + ".links.txt"), "w") as fh:
        fh.write("\n".join(sorted(links)) + ("\n" if links else ""))
    with open(os.path.join(OUT, tag + ".manifest.tsv"), "w") as fh:
        fh.write("class\tdisposition\tfiles\tbytes\n")
        for k, (n, b) in sorted(cls.items()):
            fh.write(f"{k}\t{'ARCHIVED' if k == 'record' else 'DROPPED'}\t{n}\t{b}\n")
        fh.write(f"symlinks\tRECORDED\t{len(links)}\t0\n")

    with tarfile.open(tgz, "r:gz") as tf:
        nmem = sum(1 for m in tf.getmembers() if m.isfile())
    nrec = cls["record"][0]

    return dict(tag=tag, nmem=nmem, nrec=nrec,
                kept=cls["record"][1], gz=os.path.getsize(tgz),
                dropped=sum(v[1] for kk, v in cls.items() if kk != "record"))


if __name__ == "__main__":
    ARCHIVE = sorted(d for d in os.listdir(".")
                      if d.startswith("work-pr97") and os.path.isdir(d)
                      and not os.path.islink(d))
    print(f"archiving {len(ARCHIVE)} arms -> {OUT}", flush=True)
    results = [archive_one(t) for t in ARCHIVE]

    tot_kept = tot_dropped = 0
    bad_integrity = []
    for r in results:
        tot_kept += r["kept"]
        tot_dropped += r["dropped"]
        if r["nmem"] != r["nrec"]:
            bad_integrity.append((r["tag"], r["nmem"], r["nrec"]))

    print(f"\nTOTAL archived {tot_kept/2**20:.1f} MiB (raw)  "
          f"dropped-if-removed {tot_dropped/2**30:.2f} GiB")
    print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())

    print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(ARCHIVE)} arms) ===")
    if not bad_integrity:
        print(f"PASS -- all {len(ARCHIVE)}/{len(ARCHIVE)} arms match")
        sys.exit(0)
    for tag, nmem, nrec in bad_integrity:
        print(f"  !! {tag}: tar has {nmem} members, manifest records {nrec}")
    sys.exit(1)

#!/usr/bin/env python3
"""Additive archive of the record layer of the 2026-08-03 retiring arms.

Fork of archive_records_20260802.py.  Differences:
  - reads plan.json from scripts/retire/state-20260803/ (plan_20260803.py);
  - archives the ARCHIVE set, not the whole removal set: tiers 1+2 plus the
    three uncited tier-V arms.  The other 31 tier-V arms are DROPPED whole per
    the valfast disposal contract (valfast/README.md) -- their compare summaries
    are in docs pr/23, pr/24 and pr/25;
  - output tree archive/records/pr23-25-era-20260803/<group>/.  The 08-02 tree
    pr-era-20260802/ is a scientific record and is never written into (M13);
  - an integrity gate runs per arm: tar member count must equal the manifest's
    record-file count, and the script exits non-zero if any arm fails.

For every archived arm writes <group>/<tag>.tar.gz + <tag>.links.txt +
<tag>.manifest.tsv.  Reads only; never moves, deletes or rewrites anything under
work-*.
"""
import os, re, json, tarfile, shutil, collections, sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
# Defaults reproduce the original pr/23-25 run exactly; the tidy round reuses
# this script with RETIRE_STATE / RETIRE_OUT pointed at its own dirs, so the
# 08-03 archive tree is never written into twice (M13).
SCR  = os.environ.get("RETIRE_STATE",
                      os.path.join(ROOT, "scripts", "retire", "state-20260803"))
OUT  = os.environ.get("RETIRE_OUT",
                      os.path.join(ROOT, "archive", "records", "pr23-25-era-20260803"))
os.chdir(ROOT)

plan = json.load(open(os.path.join(SCR, "plan.json")))
ARCHIVE, grp = plan["ARCHIVE"], plan["group"]
exc_path = os.path.join(SCR, "light_exceptions.txt")
FORCE = set()
if os.path.exists(exc_path):
    FORCE = {l.strip() for l in open(exc_path) if l.strip()}

HEAVY = [("pctree",  re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",    re.compile(r'^mabc.*\.zip$')),
         ("calib",   re.compile(r'^calib-evt.*\.json(\.gz)?$')),
         ("npz",     re.compile(r'.*\.npz$')),
         ("clusters",re.compile(r'^clusters-apa.*\.tar\.gz$')),
         ("opflash", re.compile(r'^opflash_apa.*\.tar\.gz$'))]
def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f): return name
    return None

tot_kept = tot_dropped = nforced = 0
bad_integrity = []
for tag in sorted(ARCHIVE):
    d = os.path.join(OUT, grp[tag]); os.makedirs(d, exist_ok=True)
    tgz = os.path.join(d, tag + ".tar.gz")
    keep, links = [], []
    cls = collections.defaultdict(lambda: [0, 0])   # class -> [n, bytes]
    for cur, sub, files in os.walk(tag):
        for name in list(sub):
            p = os.path.join(cur, name)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}"); sub.remove(name)
        for f in files:
            p = os.path.join(cur, f)
            if os.path.islink(p):
                links.append(f"{p}\t->\t{os.readlink(p)}"); continue
            try: sz = os.path.getsize(p)
            except OSError: continue
            hc = heavy_class(f)
            if hc is not None and p in FORCE:
                hc = None; nforced += 1
            cls[hc or "record"][0] += 1; cls[hc or "record"][1] += sz
            if hc is None: keep.append(p)
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(keep): tf.add(p)
    with open(os.path.join(d, tag + ".links.txt"), "w") as fh:
        fh.write("\n".join(sorted(links)) + ("\n" if links else ""))
    with open(os.path.join(d, tag + ".manifest.tsv"), "w") as fh:
        fh.write("class\tdisposition\tfiles\tbytes\n")
        for k, (n, b) in sorted(cls.items()):
            fh.write(f"{k}\t{'ARCHIVED' if k=='record' else 'DROPPED'}\t{n}\t{b}\n")
        fh.write(f"symlinks\tRECORDED\t{len(links)}\t0\n")

    # integrity gate: members in the tarball == record-class files in the manifest
    with tarfile.open(tgz, "r:gz") as tf:
        nmem = sum(1 for m in tf.getmembers() if m.isfile())
    nrec = cls["record"][0]
    if nmem != nrec:
        bad_integrity.append((tag, nmem, nrec))

    # belt and braces -- plan_20260803.py asserts there are none, but if a label
    # dir ever appears in a removal arm it is copied verbatim, never tarred (M13)
    for labname in ("nusel_labels", "ql_labels"):
        lab = os.path.join(tag, labname)
        if os.path.isdir(lab) and not os.path.islink(lab):
            dst = os.path.join(ROOT, "archive", "records", "labels", tag, labname)
            if os.path.exists(dst): shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(lab, dst, symlinks=True)
            print(f"!! unexpected {labname} in removal arm, copied verbatim: {tag}")

    k = cls["record"][1]; dr = sum(v[1] for kk, v in cls.items() if kk != "record")
    tot_kept += k; tot_dropped += dr
    print(f"{tag:38s} archived {k/2**20:8.1f} MiB -> {os.path.getsize(tgz)/2**20:7.1f} MiB gz"
          f"   drops {dr/2**20:8.1f} MiB   members {nmem}=={nrec}", flush=True)

print(f"\nTOTAL archived {tot_kept/2**20:.1f} MiB (raw)  "
      f"dropped-if-removed {tot_dropped/2**30:.2f} GiB  forced-light-files {nforced}/{len(FORCE)}")
print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())

print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(ARCHIVE)} arms) ===")
if not bad_integrity:
    print(f"PASS -- all {len(ARCHIVE)}/{len(ARCHIVE)} arms match")
    sys.exit(0)
for tag, nmem, nrec in bad_integrity:
    print(f"  !! {tag}: tar has {nmem} members, manifest records {nrec}")
sys.exit(1)

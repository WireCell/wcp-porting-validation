#!/usr/bin/env python3
"""Additive archive of the record layer of the 2026-08-05 retiring arms.

Fork of archive_records_20260803.py.  Two substantive differences and one
mechanical one:

  1. THE calib REGEX IS FIXED.  archive_records_20260803.py:41 matches
     `^calib-evt.*\\.json$`, but the pr/28-37 era writes `calib-pr-evt<N>.json`
     (pr33_cmp.py:151).  Under the old regex ~1 GiB of calib dumps across this
     round's 182 arms would be classed `record` and ARCHIVED -- contrary to
     archive/records/README.md, which lists calib dumps among the dropped
     classes, and contrary to what the plan script's footprint block reports.
     plan_20260805.py carries the same fix; the two must agree or the manifest
     and the plan disagree about what was kept.

  2. tracking-pr.root IS KEPT (it is not in HEAVY, so it lands in `record`).
     That is deliberate and it is ~1.15 GiB of this round's ~1.8 GiB archive.
     Rationale: it is one of the five artifact families pr33_cmp.py compares,
     archive/records/README.md already archives tracking-stm.root by the same
     logic, and doc pr/37 sec.2.2/sec.2.5's numbers are ROOT-leaf numbers -- with
     the arms gone (doc pr/37 sec.13.5) this is the only remaining substrate for
     re-checking them.  Owner decision 2026-08-05.

  3. 24-WAY PARALLEL.  The 08-03 round tarred serially; 182 arms is the slow step
     of this round.  Each arm is independent -- separate output paths, no shared
     state -- so it maps cleanly onto a process pool.  RETIRE_JOBS overrides.
     Output is collected and printed in sorted order so the log is deterministic
     regardless of completion order.

Output tree archive/records/pr28-37-era-20260805/<group>/.  The 08-02 and 08-03
trees are scientific records and are never written into (M13).

For every archived arm writes <group>/<tag>.tar.gz + <tag>.links.txt +
<tag>.manifest.tsv.  Reads only; never moves, deletes or rewrites anything under
work-*.
"""
import os, re, json, tarfile, shutil, collections, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.environ.get("RETIRE_STATE",
                     os.path.join(ROOT, "scripts", "retire", "state-20260805"))
OUT = os.environ.get("RETIRE_OUT",
                     os.path.join(ROOT, "archive", "records", "pr28-37-era-20260805"))
JOBS = int(os.environ.get("RETIRE_JOBS", "24"))
os.chdir(ROOT)

plan = json.load(open(os.path.join(SCR, "plan.json")))
ARCHIVE, grp = plan["ARCHIVE"], plan["group"]
exc_path = os.path.join(SCR, "light_exceptions.txt")
FORCE = set()
if os.path.exists(exc_path):
    FORCE = {l.strip() for l in open(exc_path) if l.strip()}

# NOTE 1: calib(-pr)?-evt.  tracking-pr.root is deliberately absent -> `record`.
HEAVY = [("pctree",   re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",     re.compile(r'^mabc.*\.zip$')),
         ("calib",    re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$')),
         ("npz",      re.compile(r'.*\.npz$')),
         ("clusters", re.compile(r'^clusters-apa.*\.tar\.gz$')),
         ("opflash",  re.compile(r'^opflash_apa.*\.tar\.gz$'))]


def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f):
            return name
    return None


def archive_one(tag):
    """Tar one arm's record layer.  Independent of every other arm."""
    d = os.path.join(OUT, grp[tag])
    os.makedirs(d, exist_ok=True)
    tgz = os.path.join(d, tag + ".tar.gz")
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
            if hc is not None and p in FORCE:
                hc = None
            cls[hc or "record"][0] += 1
            cls[hc or "record"][1] += sz
            if hc is None:
                keep.append(p)
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(keep):
            tf.add(p)
    with open(os.path.join(d, tag + ".links.txt"), "w") as fh:
        fh.write("\n".join(sorted(links)) + ("\n" if links else ""))
    with open(os.path.join(d, tag + ".manifest.tsv"), "w") as fh:
        fh.write("class\tdisposition\tfiles\tbytes\n")
        for k, (n, b) in sorted(cls.items()):
            fh.write(f"{k}\t{'ARCHIVED' if k=='record' else 'DROPPED'}\t{n}\t{b}\n")
        fh.write(f"symlinks\tRECORDED\t{len(links)}\t0\n")

    with tarfile.open(tgz, "r:gz") as tf:
        nmem = sum(1 for m in tf.getmembers() if m.isfile())
    nrec = cls["record"][0]

    # Belt and braces -- plan_20260805.py ASSERT 2 says there are none, but if a
    # label dir ever appears in a removal arm it is copied verbatim, never
    # tarred (M13).
    lab_note = None
    for labname in ("nusel_labels", "ql_labels"):
        lab = os.path.join(tag, labname)
        if os.path.isdir(lab) and not os.path.islink(lab):
            dst = os.path.join(ROOT, "archive", "records", "labels", tag, labname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(lab, dst, symlinks=True)
            lab_note = labname

    return dict(tag=tag, nmem=nmem, nrec=nrec, lab=lab_note,
                kept=cls["record"][1], gz=os.path.getsize(tgz),
                dropped=sum(v[1] for kk, v in cls.items() if kk != "record"))


if __name__ == "__main__":
    print(f"archiving {len(ARCHIVE)} arms -> {OUT}  ({JOBS}-way)", flush=True)
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        results = list(ex.map(archive_one, sorted(ARCHIVE)))

    tot_kept = tot_dropped = 0
    bad_integrity = []
    for r in sorted(results, key=lambda r: r["tag"]):
        tot_kept += r["kept"]
        tot_dropped += r["dropped"]
        if r["nmem"] != r["nrec"]:
            bad_integrity.append((r["tag"], r["nmem"], r["nrec"]))
        if r["lab"]:
            print(f"!! unexpected {r['lab']} in removal arm, copied verbatim: {r['tag']}")
        print(f"{r['tag']:38s} archived {r['kept']/2**20:8.1f} MiB -> "
              f"{r['gz']/2**20:7.1f} MiB gz   drops {r['dropped']/2**20:8.1f} MiB   "
              f"members {r['nmem']}=={r['nrec']}")

    print(f"\nTOTAL archived {tot_kept/2**20:.1f} MiB (raw)  "
          f"dropped-if-removed {tot_dropped/2**30:.2f} GiB  forced-light-files {len(FORCE)}")
    print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())

    print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(ARCHIVE)} arms) ===")
    if not bad_integrity:
        print(f"PASS -- all {len(ARCHIVE)}/{len(ARCHIVE)} arms match")
        sys.exit(0)
    for tag, nmem, nrec in bad_integrity:
        print(f"  !! {tag}: tar has {nmem} members, manifest records {nrec}")
    sys.exit(1)

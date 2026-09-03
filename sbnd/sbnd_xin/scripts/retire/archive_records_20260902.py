#!/usr/bin/env python3
"""Archive the record layer of the 2026-09-02 retiring arms -- the doc 94
STM rounds, the doc 97 separation A/B, and the superseded production epoch.

Fork of archive_records_20260901c.py by verbatim `cp` (cmp-verified before
editing).  Only the header, RETIRE_STATE and RETIRE_OUT defaults changed; the
HEAVY classification and the integrity gate are untouched, deliberately -- they
were correct for the 09-01 round and this round retires arms of exactly the
same two shapes (stage-A ql_evt<N>/ and stage-B pr_evt<N>/).

WHAT THE RECORD LAYER IS HERE.  The heavy classes (pctree, mabc, calib, npz,
clusters, opflash, tracking, groupin) are dropped; what is archived is
stdout.log + wct_{ql,pr}_evt<N>.log + .wct-cfg-evt<N>.json + nusel-evt<N>.tsv
+ the arm-level nusel-events.tsv / nusel-table.tsv.  That is what every doc 94
and doc 97 claim actually rests on -- the per-event verdict tables and the
compiled config each arm ran under -- so the claims stay re-checkable after
the bytes are gone.

SYMLINKS ARE RECORDED, NOT FOLLOWED.  This matters more this round than last:
the retiring stage-A arms (d97on, d97off2) are ~50%% symlink by count, 2000
links each into work-mcp2k-grp0825/evt<N>.  archive_one() removes link dirs
from the walk and writes them to <tag>.links.txt, so no record tar can pull in
a copy of the imaging substrate.

Output tree archive/records/campaign-close-20260902/<group>/.  Never writes
into an earlier round's archive tree (M13).
"""
import os, re, json, tarfile, shutil, collections, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.environ.get("RETIRE_STATE",
                     os.path.join(ROOT, "scripts", "retire", "state-20260902"))
OUT = os.environ.get("RETIRE_OUT",
                     os.path.join(ROOT, "archive", "records", "campaign-close-20260902"))
JOBS = int(os.environ.get("RETIRE_JOBS", "12"))
os.chdir(ROOT)

plan = json.load(open(os.path.join(SCR, "plan.json")))
ARCHIVE, grp = plan["ARCHIVE"], plan["group"]

HEAVY = [("pctree",   re.compile(r'^pctree.*\.tar\.gz$')),
         ("mabc",     re.compile(r'^mabc.*\.zip$')),
         ("calib",    re.compile(r'^calib(-pr)?-evt.*\.json(\.gz)?$')),
         ("npz",      re.compile(r'.*\.npz$')),
         ("clusters", re.compile(r'^clusters-apa.*\.tar\.gz$')),
         ("opflash",  re.compile(r'^opflash_apa.*\.tar\.gz$')),
         ("tracking", re.compile(r'^tracking-pr\.root$')),
         ("oc56scan", re.compile(r'^oc56scan-evt.*\.jsonl$'))]
# full-path classes (see the header): the group INPUT archive of a group-mode
# PR run, proven duplicate of a surviving grp0825 Q/L root.
HEAVY_PATH = [("groupin", re.compile(r'(^|/)\.groups/g\d+\.tar\.gz$'))]


def heavy_class(f, path=None):
    for name, pat in HEAVY:
        if pat.match(f):
            return name
    if path is not None:
        for name, pat in HEAVY_PATH:
            if pat.search(path):
                return name
    return None


def archive_one(tag):
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
            hc = heavy_class(f, p)
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

    # Belt and braces -- ASSERT 2 says there are none, but if a label dir ever
    # appears in a removal arm, copy it verbatim, never tar it (M13).
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
                groupin=cls["groupin"][1],
                dropped=sum(v[1] for kk, v in cls.items() if kk != "record"))


if __name__ == "__main__":
    print(f"archiving {len(ARCHIVE)} arms -> {OUT}  ({JOBS}-way)", flush=True)
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        results = list(ex.map(archive_one, sorted(ARCHIVE)))

    tot_kept = tot_dropped = tot_gz = tot_gin = 0
    bad_integrity = []
    for r in sorted(results, key=lambda r: r["tag"]):
        tot_kept += r["kept"]
        tot_dropped += r["dropped"]
        tot_gz += r["gz"]
        tot_gin += r["groupin"]
        if r["nmem"] != r["nrec"]:
            bad_integrity.append((r["tag"], r["nmem"], r["nrec"]))
        if r["lab"]:
            print(f"!! unexpected {r['lab']} in removal arm, copied verbatim: {r['tag']}")

    print(f"\nTOTAL archived {tot_kept/2**30:.2f} GiB raw -> {tot_gz/2**30:.2f} GiB gz  "
          f"dropped-if-removed {tot_dropped/2**30:.2f} GiB")
    print(f"  of which the new groupin class dropped {tot_gin/2**30:.2f} GiB "
          f"(proven duplicates, state-20260901c/group-dupes.tsv)")
    print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())

    print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(ARCHIVE)} arms) ===")
    if not bad_integrity:
        print(f"PASS -- all {len(ARCHIVE)}/{len(ARCHIVE)} arms match")
        sys.exit(0)
    for tag, nmem, nrec in bad_integrity:
        print(f"  !! {tag}: tar has {nmem} members, manifest records {nrec}")
    sys.exit(1)

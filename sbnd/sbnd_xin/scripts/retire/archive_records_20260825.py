#!/usr/bin/env python3
"""Archive of the record layer of the 2026-08-25 retiring arms -- the doc-81
group-mode re-baseline releasing the pr/104 + pr/112 option scan.

Fork of archive_records_20260823.py.  HEAVY is carried unchanged, and this
time that is measured rather than argued: a census of all 66 removal arms
(72.6 GiB) finds ZERO unclassified file above 5 MiB, so nothing heavy can slip
into the record tar.  The record layer is 8.2 GiB raw and is dominated by
20891 .wct-cfg-evt<N>.json (266 KB each) -- the compiled config each arm
actually ran, i.e. its operating point, which is exactly the thing worth
keeping and compresses hard.  The integrity gate below still catches a
tar/manifest mismatch if a class was missed.

Output tree archive/records/prod0825-groupmode-20260825/<group>/.  Never writes
into an earlier round's archive tree (M13).

For every archived arm writes <group>/<tag>.tar.gz + <tag>.links.txt +
<tag>.manifest.tsv.  Reads only; never moves, deletes or rewrites anything
under work-*.
"""
import os, re, json, tarfile, shutil, collections, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.environ.get("RETIRE_STATE",
                     os.path.join(ROOT, "scripts", "retire", "state-20260825"))
OUT = os.environ.get("RETIRE_OUT",
                     os.path.join(ROOT, "archive", "records", "prod0825-groupmode-20260825"))
JOBS = int(os.environ.get("RETIRE_JOBS", "24"))
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


def heavy_class(f):
    for name, pat in HEAVY:
        if pat.match(f):
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
            hc = heavy_class(f)
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

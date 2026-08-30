#!/usr/bin/env python3
"""Archive the record layer of the 2026-08-29 retiring arms -- eleven closed
doc families (pr/117-pr/128, doc 84 r1-r4, doc 80's MCS arms, doc 114's
display arms) released while the pi0 epoch and the open pr/130 round stay.

Fork of archive_records_20260825.py.  ONE class is added to HEAVY and that
addition is the interesting part of this round's archive step.

HEAVY GAINED A CLASS, MEASURED RATHER THAN ARGUED.  The 08-25 round justified
reusing HEAVY unchanged with a census of its 66 removal arms: "ZERO
unclassified file above 5 MiB, so nothing heavy can slip into the record tar".
That statement was true of its removal set and is FALSE of this one, for a
reason no one could have carried forward by memory: none of its arms was
group-mode, and doc 84 round 3's census arms are.  This round's census found
242 unclassified files above 5 MiB, all of them .groups/g<N>.tar.gz -- 188
files, 4.96 GiB -- the group INPUT archives, i.e. bundles of Q/L pctrees fed to
a group-mode PR run.  Copied data, not a record.

Left unclassified they would have gone verbatim into the record tar, tripling
the record layer to 16.06 GiB for nothing.  So the class is DROPPED -- but only
after verify_group_dupes_20260829.py proved every one of the 188 is a pure
duplicate of a Q/L root that SURVIVES this round: 1231656/1231656 members
byte-identical to work-{mcp1k,mcp2k}-grp0825, member by member, recorded in
state-20260829/group-dupes.tsv.  Run that first; the driver's interlock 4
checks its row count before deleting anything.

Note the matching rule: HEAVY_PATH matches the FULL PATH `.groups/g<N>.tar.gz`,
not the basename.  heavy_class() takes both.  A stray g1.tar.gz somewhere else
in a tree is not a group input archive and stays a record.

Everything else is carried: the record layer is dominated by per-event
stdout.log + wct_pr_evt<N>.log (16488 each) and .wct-cfg-evt<N>.json (13487,
266 KB each -- the compiled config each arm actually ran, i.e. its operating
point, the one thing most worth keeping and the thing that compresses hardest).
The integrity gate below still catches a tar/manifest mismatch if a class was
missed.

Output tree archive/records/em-pr-era-20260829/<group>/.  Never writes into an
earlier round's archive tree (M13).

For every archived arm writes <group>/<tag>.tar.gz + <tag>.links.txt +
<tag>.manifest.tsv.  Reads only; never moves, deletes or rewrites anything
under work-*.
"""
import os, re, json, tarfile, shutil, collections, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
SCR = os.environ.get("RETIRE_STATE",
                     os.path.join(ROOT, "scripts", "retire", "state-20260829"))
OUT = os.environ.get("RETIRE_OUT",
                     os.path.join(ROOT, "archive", "records", "em-pr-era-20260829"))
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
          f"(proven duplicates, state-20260829/group-dupes.tsv)")
    print("archive dir size:", os.popen(f"du -sh {OUT}").read().strip())

    print(f"\n=== INTEGRITY GATE (tar members == manifest record files, {len(ARCHIVE)} arms) ===")
    if not bad_integrity:
        print(f"PASS -- all {len(ARCHIVE)}/{len(ARCHIVE)} arms match")
        sys.exit(0)
    for tag, nmem, nrec in bad_integrity:
        print(f"  !! {tag}: tar has {nmem} members, manifest records {nrec}")
    sys.exit(1)

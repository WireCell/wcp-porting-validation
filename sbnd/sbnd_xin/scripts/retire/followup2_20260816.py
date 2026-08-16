#!/usr/bin/env python3
"""Follow-up to the 2026-08-16 round, same day: owner asked to reduce work*
further, to "only the latest one, as well as input to achieve it" per sample.

pr87ion3 is the latest for mcp1k/nuecc48/ncpi0; its input is the cb0805/img
hubs, already KEEP. That makes three families no longer "latest or input" and
they retire here:

  work-{mcp1k,nuecc48,ncpi0}-prod0813   the prior production reference
  work-{mcp1k,nuecc48,ncpi0}-ma10       the DL-vertex deployed-baseline arm
  work-pr87-postflip-{mcp1k,nuecc48,ncpi0}  the byte-identity proof arm

r1qlmc/r2mc are UNCHANGED: prod0813 IS their latest (no pr83/85/86/87 arm
exists for either), and both are live on bokeh :5017/:5018 right now.

Owner-confirmed trade-off (asked explicitly before running this):
  - vertex_labels/ (481 of 483 labels) has `source` inside a -prod0813 dump
    for these three samples.  The label JSON itself is self-contained (already
    established precedent: uitest75/vtxscan1 labels stay readable with their
    arm already gone) but the underlying dump becomes unopenable -- no more
    re-rendering pr85_panels2d.py / pr86_kink_census.py against these events'
    actual PR result.
  - baselines.py's deployed_dump_path() stops resolving -- the 473-event
    DL-vertex analysis set (docs pr/77-81) dies as a re-derivable input;
    the already-built dl_vtx_training/data/* snapshots (kept, 20M) still back
    every number those docs quote.

Safety checks run and PASSED before this script exists:
  - 0 git-tracked files inside any of the 9 dirs
  - 0 nusel_labels/ql_labels dirs inside any of the 9 dirs
  - 0 symlinks anywhere in the tree point into any of the 9 dirs
  - both live bokeh viewers (:5017, :5018) reference only r1qlmc/r2mc-prod0813

Same archive-then-delete pattern as Phase 2: record layer (logs, rc.txt,
.time.meta, nusel-evt*.tsv, tables) archived to
archive/records/pr79-86-era-20260816/<group>/ before rm -rf; HEAVY classes
(calib json, pctree tar, mabc zip, tracking root) dropped, never archived --
they are the exact reference-arm PR products this follow-up removes.

  python3 followup2_20260816.py            # dry run
  CONFIRM=yes python3 followup2_20260816.py  # archive + delete
"""
import os
import re
import shutil
import sys
import tarfile

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
OUT = os.path.join(ROOT, "archive", "records", "pr79-86-era-20260816")
STATE = os.path.join(ROOT, "scripts", "retire", "state-20260816")
MANIFEST = os.path.join(STATE, "followup2-removed.tsv")
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"
os.chdir(ROOT)

DROP = {
    "work-mcp1k-prod0813":         "prod0813-arms",
    "work-nuecc48-prod0813":       "prod0813-arms",
    "work-ncpi0-prod0813":         "prod0813-arms",
    "work-mcp1k-ma10":             "ma10-family",
    "work-nuecc48-ma10":           "ma10-family",
    "work-ncpi0-ma10":             "ma10-family",
    "work-pr87-postflip-mcp1k":    "pr87-postflip",
    "work-pr87-postflip-nuecc48":  "pr87-postflip",
    "work-pr87-postflip-ncpi0":    "pr87-postflip",
}

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


for d in DROP:
    if not os.path.isdir(d):
        sys.exit(f"REFUSING: {d} not found -- already gone or a naming mismatch")

if CONFIRM and os.path.exists(MANIFEST):
    sys.exit(f"REFUSING: {MANIFEST} exists -- already run this round (M13)")

pre_broken = int(os.popen(f"find {ROOT} -xtype l | wc -l").read().strip())
print(f"broken symlinks before: {pre_broken}   (MUST be 0)")
if pre_broken != 0:
    sys.exit("REFUSING: tree already has dangling links")

viewers = os.popen("pgrep -af 'bokeh serve'").read()
hit = [d for d in DROP if d in viewers]
if hit:
    sys.exit(f"REFUSING: a live viewer references {hit}")
print("viewer check: clean (r1qlmc/r2mc-prod0813 only, both untouched)")

total_mb = 0
for d, grp in sorted(DROP.items()):
    sz = sum(os.path.getsize(os.path.join(c, f))
             for c, _, fs in os.walk(d) for f in fs
             if not os.path.islink(os.path.join(c, f)))
    total_mb += sz / 2**20
    print(f"  {'would archive+remove' if not CONFIRM else 'archiving':22s} {d:32s} {sz/2**20:8.0f} MB  [{grp}]")
print(f"total: {total_mb/1024:.2f} GiB")

if not CONFIRM:
    print("\ndry run only -- re-run with CONFIRM=yes to archive + delete")
    sys.exit(0)

os.makedirs(STATE, exist_ok=True)
rows = []
for d, grp in sorted(DROP.items()):
    gdir = os.path.join(OUT, grp)
    os.makedirs(gdir, exist_ok=True)
    tgz = os.path.join(gdir, d + ".tar.gz")
    keep, links = [], []
    kept_bytes = dropped_bytes = 0
    for cur, sub, files in os.walk(d):
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
            sz = os.path.getsize(p)
            if heavy_class(f) is None:
                keep.append(p)
                kept_bytes += sz
            else:
                dropped_bytes += sz
    with tarfile.open(tgz, "w:gz") as tf:
        for p in sorted(keep):
            tf.add(p)
    with open(os.path.join(gdir, d + ".links.txt"), "w") as fh:
        fh.write("\n".join(sorted(links)) + ("\n" if links else ""))
    with tarfile.open(tgz, "r:gz") as tf:
        nmem = sum(1 for m in tf.getmembers() if m.isfile())
    if nmem != len(keep):
        sys.exit(f"REFUSING: integrity mismatch for {d}: tar has {nmem}, expected {len(keep)}")
    print(f"  OK archived+verified  {d}  ({nmem} record files, {kept_bytes/2**20:.1f} MiB -> {tgz})")

    sz_total = kept_bytes + dropped_bytes
    mt = os.popen(f"date -Is -r {d}").read().strip()
    shutil.rmtree(d)
    print(f"  removed  {d}  ({sz_total/2**20:.0f} MB)")
    rows.append((d, grp, sz_total, os.path.basename(tgz), mt))

with open(MANIFEST, "w") as fh:
    fh.write("# followup2_20260816.py -- latest+input trim, owner-confirmed\n")
    fh.write(f"# wcp-porting-img HEAD {os.popen('git -C ' + ROOT + ' rev-parse --short HEAD').read().strip()}\n")
    fh.write("dir\tgroup\tbytes\tarchive_tarball\tdir_mtime\n")
    for d, grp, sz, tgz, mt in rows:
        fh.write(f"{d}\t{grp}\t{sz}\t{tgz}\t{mt}\n")

post_broken = int(os.popen(f"find {ROOT} -xtype l | wc -l").read().strip())
print(f"\nbroken symlinks after: {post_broken}   (MUST be 0)")
print("manifest:", MANIFEST)

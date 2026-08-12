#!/usr/bin/env python3
"""Phase 4 of the 2026-08-11 retirement round: thin the five surviving `-cb0805`
Q/L hubs and drop work-img-mcp1k's regenerable SP-frame layer.

The Phase 3 sweep (retire_20260811.sh) only ever touched the 454-dir removal
set; it never modified a KEEP hub. This script does the second, hub-internal
cut the plan called for:

  - Inside work-{mcp1k,nuecc48,ncpi0,r1qlmc,r2mc}-cb0805: the `pr_evt*/` and
    `nusel_evt*/` layers are 2026-08-05 PR/nusel output, entirely superseded
    by the current-production `work-pr64r4-{on48,on19,on1k}` reference arms
    (which cover the frozen Bee event lists) -- the PR chain and Bee builder
    never read stale pr_evt/nusel_evt output from a Q/L root, only ql_evt*/.
    Precedent: the 2026-08-03 round's tier C (calib dumps). ~8.8 GiB.
  - Every `ql_evt*/calib-evt*.json` dump alongside them -- `run_pr_chain_batch.sh`
    and `make_pr_bee.py` never read it; it is only consumed when the Q/L step
    itself re-runs.
  - work-img-mcp1k/evt*/sp-frames*.tar.bz2 (1.23 GiB) -- regenerable from
    input_files_reco1/staged-mcp2025c-1000evt (1004 source entries present,
    checked below), and not read by anything downstream of imaging.

The `ql_evt*/` layer itself (pctree-evt*.tar.gz, mabc-all-apa.zip,
opflash_apa*.tar.gz, wct_ql_evt*.log, icluster-*.npz) is NEVER touched -- it is
what run_pr_chain_batch.sh and make_pr_bee.py read, and what the 1090 inbound
symlinks from the -img- hubs resolve into.

HARD REFUSALS (this script only ever deletes inside these five named roots
plus work-img-mcp1k, and only these specific patterns):
  - any candidate path not under one of the six named roots
  - any path inside `ql_evt*/` other than a `calib-evt*.json` file
  - any symlink (this round never deletes a symlink; a hub's ql_evt*/icluster
    files ARE symlinks into work-img-*, and must not be touched here)

Record layer archived FIRST, to archive/records/pr38-65-era-20260811/cb0805-hubs/
<root>.tar.gz: wct_pr_evt*.log, wct_nusel_evt*.log, nusel-evt*.tsv,
tracking-stm.root, rc.txt, stdout.log, .time.meta, plus the root-level
nusel-table.tsv/nusel-events.tsv.

  python3 thin_hubs_20260811.py           # dry run (default)
  CONFIRM=yes python3 thin_hubs_20260811.py
"""
import os, re, sys, tarfile, shutil

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
os.chdir(ROOT)
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"

CB0805_HUBS = ["work-mcp1k-cb0805", "work-nuecc48-cb0805", "work-ncpi0-cb0805",
               "work-r1qlmc-cb0805", "work-r2mc-cb0805"]
ALLOWED_ROOTS = set(CB0805_HUBS) | {"work-img-mcp1k"}

OUT = os.path.join(ROOT, "archive", "records", "pr38-65-era-20260811", "cb0805-hubs")
os.makedirs(OUT, exist_ok=True)

RECORD_PAT = re.compile(
    r'^(wct_pr_evt.*\.log|wct_nusel_evt.*\.log|nusel-evt.*\.tsv|tracking-stm\.root|'
    r'rc\.txt|stdout\.log|\.time\.meta)$')


def guard(path, root):
    """Every deletion goes through here. Refuse anything outside the contract."""
    if root not in ALLOWED_ROOTS:
        raise SystemExit(f"REFUSE: root {root} not in ALLOWED_ROOTS")
    if os.path.islink(path):
        raise SystemExit(f"REFUSE: symlink {path} -- this script never deletes a symlink")
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts[0].startswith('ql_evt'):
        if len(parts) != 2 or not re.match(r'^calib-evt.*\.json$', parts[1]):
            raise SystemExit(f"REFUSE: {path} is inside ql_evt* but is not a calib-evt*.json file")
    return True


fail = 0
total_archived = total_dropped = 0

print("=== ASSERT: input_files_reco1/staged-mcp2025c-1000evt covers 1000 events ===")
src = "input_files_reco1/staged-mcp2025c-1000evt"
n = len(os.listdir(src)) if os.path.isdir(src) and not os.path.islink(src) else 0
ok = n >= 1000
print(f"  {'OK ' if ok else '!! '} {src}  ({n} entries, need >=1000)")
if not ok:
    fail += 1

for hub in CB0805_HUBS:
    print(f"\n=== {hub} ===")
    if not os.path.isdir(hub):
        print("  MISSING -- skip")
        continue
    tgz = os.path.join(OUT, hub + ".tar.gz")
    record_files, drop_dirs = [], []
    n_heavy_files = 0
    heavy_bytes = 0
    for entry in sorted(os.listdir(hub)):
        p = os.path.join(hub, entry)
        if os.path.islink(p):
            continue
        if entry in ("nusel-table.tsv", "nusel-events.tsv") and os.path.isfile(p):
            record_files.append(p)
            continue
        if not os.path.isdir(p):
            continue
        if not (entry.startswith("pr_evt") or entry.startswith("nusel_evt")):
            continue
        guard(p, hub)
        for cur, sub, files in os.walk(p):
            for f in files:
                fp = os.path.join(cur, f)
                if os.path.islink(fp):
                    continue
                sz = os.path.getsize(fp)
                if RECORD_PAT.match(f):
                    record_files.append(fp)
                else:
                    n_heavy_files += 1
                    heavy_bytes += sz
        drop_dirs.append(p)

    # calib dumps inside ql_evt*/ -- record layer does NOT include these (they
    # are heavy per the plan's classification, dropped without archiving,
    # same as the 2026-08-03 tier C precedent).
    calib_files = []
    calib_bytes = 0
    for entry in sorted(os.listdir(hub)):
        if not entry.startswith("ql_evt"):
            continue
        qp = os.path.join(hub, entry)
        if os.path.islink(qp) or not os.path.isdir(qp):
            continue
        for f in os.listdir(qp):
            if re.match(r'^calib-evt.*\.json$', f):
                fp = os.path.join(qp, f)
                if os.path.islink(fp):
                    continue
                guard(fp, hub)
                calib_files.append(fp)
                calib_bytes += os.path.getsize(fp)

    print(f"  record files to archive: {len(record_files)}")
    print(f"  pr_evt*/nusel_evt* dirs to remove: {len(drop_dirs)}  "
          f"({heavy_bytes/2**20:.1f} MiB heavy)")
    print(f"  ql_evt*/calib-evt*.json to remove: {len(calib_files)}  "
          f"({calib_bytes/2**20:.1f} MiB)")
    total_dropped += heavy_bytes + calib_bytes

    if CONFIRM:
        with tarfile.open(tgz, "w:gz") as tf:
            for p in record_files:
                tf.add(p)
        with tarfile.open(tgz, "r:gz") as tf:
            nmem = sum(1 for m in tf.getmembers() if m.isfile())
        if nmem != len(record_files):
            print(f"  !! INTEGRITY FAIL: tar has {nmem}, expected {len(record_files)}")
            fail += 1
            continue
        print(f"  archived -> {tgz} ({os.path.getsize(tgz)/2**20:.1f} MiB gz), "
              f"integrity {nmem}=={len(record_files)}")
        total_archived += sum(os.path.getsize(p) for p in record_files)
        for p in drop_dirs:
            shutil.rmtree(p)
        for p in calib_files:
            os.remove(p)
        print(f"  removed {len(drop_dirs)} dirs + {len(calib_files)} calib files")

print("\n=== work-img-mcp1k sp-frames*.tar.bz2 ===")
hub = "work-img-mcp1k"
sp_files, sp_bytes = [], 0
if os.path.isdir(hub):
    for entry in sorted(os.listdir(hub)):
        p = os.path.join(hub, entry)
        if os.path.islink(p) or not os.path.isdir(p) or not entry.startswith("evt"):
            continue
        for f in os.listdir(p):
            if re.match(r'^sp-frames.*\.tar\.bz2$', f):
                fp = os.path.join(p, f)
                if os.path.islink(fp):
                    continue
                guard(fp, hub)
                sp_files.append(fp)
                sp_bytes += os.path.getsize(fp)
print(f"  {len(sp_files)} files, {sp_bytes/2**20:.1f} MiB")
total_dropped += sp_bytes
if CONFIRM and ok:
    for p in sp_files:
        os.remove(p)
    print(f"  removed {len(sp_files)} sp-frames files")
elif CONFIRM and not ok:
    print("  !! SKIPPED -- source coverage assert failed above")

print(f"\nTOTAL {'archived' if CONFIRM else 'would archive'} "
      f"{total_archived/2**20:.1f} MiB, "
      f"{'dropped' if CONFIRM else 'would drop'} {total_dropped/2**30:.2f} GiB")
if not CONFIRM:
    print("\nDRY RUN -- re-run with CONFIRM=yes to execute")
sys.exit(0 if not fail else 1)

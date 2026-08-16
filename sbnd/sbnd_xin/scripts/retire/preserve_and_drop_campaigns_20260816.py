#!/usr/bin/env python3
"""Phase 4, 2026-08-16: preserve the 22 nusel_labels/ trees inside the three
old campaign archives (archive/{tgm-docs29-39,aborted-d54,stm-docs40-49}),
verify byte-identity, then drop the three archives whole (1.25 GiB).

Copy-then-verify-then-remove, modelled on preserve_20260805cs.sh. Never
removes anything until every label dir has a verified byte-identical copy
under archive/records/labels/<campaign>/<tag>/nusel_labels.

  python3 preserve_and_drop_campaigns_20260816.py            # dry run
  CONFIRM=yes python3 preserve_and_drop_campaigns_20260816.py  # copy + verify + drop
"""
import filecmp
import os
import shutil
import sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
CAMPAIGNS = ["tgm-docs29-39", "aborted-d54", "stm-docs40-49"]
LABROOT = os.path.join(ROOT, "archive", "records", "labels")
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"
os.chdir(ROOT)


def tree_identical(a, b):
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(tree_identical(os.path.join(a, d), os.path.join(b, d))
               for d in cmp.common_dirs)


# ---- find every nusel_labels tree inside the three campaigns ---------------
sources = []
for camp in CAMPAIGNS:
    camp_dir = os.path.join("archive", camp)
    if not os.path.isdir(camp_dir):
        continue
    for cur, sub, files in os.walk(camp_dir):
        if os.path.basename(cur) == "nusel_labels":
            sources.append((camp, cur))

print(f"found {len(sources)} nusel_labels/ trees across {len(CAMPAIGNS)} campaigns")
for camp, src in sources:
    print(f"  {src}")

# ---- copy step ---------------------------------------------------------
copied = []
for camp, src in sources:
    tag = os.path.relpath(src, os.path.join("archive", camp))  # e.g. work-mcp10-lm2/nusel_labels
    dst = os.path.join(LABROOT, camp, tag)
    if not CONFIRM:
        print(f"  would copy  {src}  ->  {os.path.relpath(dst, ROOT)}")
        continue
    if os.path.isdir(dst):
        if tree_identical(src, dst):
            print(f"  OK (already present, identical)  {os.path.relpath(dst, ROOT)}")
            copied.append((src, dst))
            continue
        sys.exit(f"REFUSING: {dst} exists and DIFFERS from {src} -- manual resolution needed")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    if not tree_identical(src, dst):
        sys.exit(f"REFUSING: copy verification FAILED for {src} -- stopping, nothing removed")
    print(f"  OK copied+verified  {os.path.relpath(dst, ROOT)}")
    copied.append((src, dst))

if not CONFIRM:
    print("\ndry run only -- re-run with CONFIRM=yes to copy, verify, and drop the archives")
    sys.exit(0)

if len(copied) != len(sources):
    sys.exit(f"REFUSING: only {len(copied)}/{len(sources)} label trees verified -- not dropping anything")

print(f"\nall {len(copied)} label trees verified byte-identical under archive/records/labels/")

# ---- drop the three campaign archives ----------------------------------
for camp in CAMPAIGNS:
    camp_dir = os.path.join("archive", camp)
    if not os.path.isdir(camp_dir):
        print(f"  SKIP (already gone): {camp_dir}")
        continue
    sz = sum(os.path.getsize(os.path.join(c, f))
             for c, _, fs in os.walk(camp_dir) for f in fs
             if not os.path.islink(os.path.join(c, f)))
    shutil.rmtree(camp_dir)
    print(f"  removed  {camp_dir}  ({sz/2**20:.0f} MB)")

print("\nPhase 4 done.")

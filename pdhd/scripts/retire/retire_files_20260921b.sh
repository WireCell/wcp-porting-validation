#!/bin/bash
# Cleanup ROUND J 2026-09-21 (doc sbnd_xin/122) -- FILE-level delete driver.
# DRY RUN unless CONFIRM=yes.
#
#   ./retire_files_20260921b.sh
#   CONFIRM=yes ./retire_files_20260921b.sh
#
# Owner, 2026-09-21, choosing from a measured menu: the pctree release, the SP frames release,
# the two superseded arms and the compression pass.  This driver covers the first two.
#   sbnd-pctree-closed  10036 files  21.27 GiB   the two CLOSED stage-B comparison arms
#   pdhd-sp-frames        128 files   7.24 GiB   unshared SP output only (see DEFERRED below)
#   sbnd-sp-frames        636 files  10.30 GiB   d115 + d102m frames-dnn
#
# ORDER: plan_files_20260921b.py --hash (lists + SHA-256 freeze) -> this.
# Gates, each with its own exit code -- the round-D sec 14 shape, plus what round J added:
#   2   no plan list for a set
#   3   N of M targets already gone (stale list; re-plan)
#   4   a target is a symlink or not a regular file
#   5   a target is outside its set's allow-list, or under an input/record/reference tree
#   6   the record dir is missing -- run plan_files_20260921b.py --hash first
#   10  the confirm-time re-plan fails a gate (F1-F7)
#   11  the re-planned lists differ from the plan-time lists (something moved)
#   14  record gate: a target has no manifest row carrying its own size AND sha256
#   16  the broken-symlink count of a tree ROSE after the delete
#
# DEFERRED, and NOT in any list here: 120 pdhd frame archives (6.77 GiB) that a symlink in a
# kept arm resolves onto (029107_*_d09ctl, _stm0 ...).  Those arms hold the frames by reference
# so they are not duplicated; releasing one degrades several kept arms at once and leaves
# dangling links inside PROTECTED directories.  That needs its own owner call.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260921b
SETS="sbnd-pctree-closed pdhd-sp-frames sbnd-sp-frames"
CONFIRM=${CONFIRM:-no}
REC=${REC_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}}

for s in $SETS; do
  [ -s "$D/files_${s}_${STAMP}.txt" ] \
    || { echo "REFUSING: no plan list for $s -- run plan_files_${STAMP}.py"; exit 2; }
done

# pre-count broken symlinks per tree so gate 16 means something (INTERLOCK 4's rule)
for t in pdvd/work pdhd/work sbnd/sbnd_xin; do
  k=$(echo "$t" | tr / _)
  find "$R/$t" -xtype l 2>/dev/null | wc -l > "$D/prebroken_files_${k}_${STAMP}.txt"
done

if [ "$CONFIRM" = yes ]; then
  echo "== re-plan at confirm time"
  PLAN_SUFFIX=confirm python3 "$D/plan_files_${STAMP}.py" \
      > "$D/plan_files_${STAMP}.confirm.out" 2>&1 \
    || { echo "REFUSING: a gate fails on re-plan ($D/plan_files_${STAMP}.confirm.out)"; exit 10; }
  for s in $SETS; do
    cmp -s "$D/files_${s}_${STAMP}.txt" "$D/files_${s}_${STAMP}.confirm.txt" \
      || { echo "REFUSING: the $s list moved between plan and confirm"; exit 11; }
  done
  echo "   OK: F1-F7 PASS, lists unchanged"
fi

python3 - "$D" "$STAMP" "$REC" "$SETS" "$CONFIRM" "$R" <<'PY' || exit $?
import os, sys
D, STAMP, REC, SETS, CONFIRM, R = sys.argv[1:7]
S = f"{R}/sbnd/sbnd_xin"
ALLOW = {
 "sbnd-pctree-closed": [f"{S}/work-{s}-{a}" for s in ("r3cv","r3nue","r3off")
                                            for a in ("d115pr","d116s0rep")],
 "pdhd-sp-frames":     [f"{R}/pdhd/work"],
 "sbnd-sp-frames":     [f"{S}/work-{s}-d115" for s in ("r3cv","r3nue","r3off")] +
                       [f"{S}/work-{s}-d102m" for s in ("mcp1k","mcp2k","ncpi0","nuecc48")],
}
FORBIDDEN = ("input_files_reco1", "xin-round3-samples", "input_data_", "/archive/",
             "/products/", "/ref/", "/docs/", "/scripts/")
rc = 0
for s in SETS.split():
    files = [l.strip() for l in open(f"{D}/files_{s}_{STAMP}.txt") if l.strip()]
    gone  = [f for f in files if not os.path.lexists(f)]
    if gone:
        print(f"REFUSING: {len(gone)} of {len(files)} {s} targets already gone -- re-plan"); sys.exit(3)
    bad = [f for f in files if os.path.islink(f) or not os.path.isfile(f)]
    if bad:
        print(f"REFUSING: {len(bad)} {s} targets are not regular files, e.g. {bad[0]}"); sys.exit(4)
    allow = [os.path.realpath(a) for a in ALLOW[s]]
    off = [f for f in files
           if any(k in os.path.realpath(f) for k in FORBIDDEN)
           or not any(os.path.realpath(f).startswith(a + "/") for a in allow)]
    if off:
        print(f"REFUSING: {len(off)} {s} targets are outside the allow-list, e.g. {off[0]}"); sys.exit(5)
    man = f"{REC}/{s}.manifest.tsv"
    if not os.path.exists(man):
        print(f"REFUSING: no record for {s} ({man}) -- run plan_files_{STAMP}.py --hash first"); sys.exit(6)
    rows = {}
    for line in open(man):
        if line.startswith("#"): continue
        p, size, digest = line.rstrip("\n").split("\t")
        rows[os.path.join(R, p)] = (int(size), digest)
    missing = [f for f in files
               if f not in rows or rows[f][0] != os.path.getsize(f) or len(rows[f][1]) != 64]
    if missing:
        print(f"REFUSING: {len(missing)} of {len(files)} {s} targets lack a manifest row with "
              f"their own size and a sha256, e.g. {missing[0]}"); sys.exit(14)
    tot = sum(os.path.getsize(f) for f in files)
    print(f"== {s}: {len(files)} files, {tot/2**30:.2f} GiB, frozen 1:1 in {os.path.basename(man)}")
    if CONFIRM != "yes":
        print(f"   DRY RUN -- nothing removed.  e.g. {os.path.relpath(files[0], R)}")
        continue
    n = 0
    for f in files:
        try: os.unlink(f); n += 1
        except OSError as e:
            print(f"   FAILED {f}: {e}"); rc = 1
    print(f"   removed {n}/{len(files)}")
sys.exit(rc)
PY

if [ "$CONFIRM" = yes ]; then
  for t in pdvd/work pdhd/work sbnd/sbnd_xin; do
    k=$(echo "$t" | tr / _)
    before=$(cat "$D/prebroken_files_${k}_${STAMP}.txt")
    after=$(find "$R/$t" -xtype l 2>/dev/null | wc -l)
    echo "== $t broken symlinks: $before -> $after"
    [ "$after" -le "$before" ] || { echo "REFUSING (after the fact): $t broken-symlink count ROSE"; exit 16; }
  done
fi
echo "done (CONFIRM=$CONFIRM)"

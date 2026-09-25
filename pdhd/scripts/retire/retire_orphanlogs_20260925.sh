#!/bin/bash
# Cleanup ROUND I 2026-09-21 (doc sbnd_xin/121 sec 5) -- ORPHANED DRIVER LOG delete driver.
# DRY RUN unless CONFIRM=yes.
#
#   ./retire_orphanlogs_20260925.sh
#   CONFIRM=yes ./retire_orphanlogs_20260925.sh
#
# Owner, 2026-09-21, asked with the measurement in hand: "Yes, orphans only."
# The set is `<tree>/work/.batch_pr_<run6>_<evt>[_<arm>].log` whose event directory no
# longer exists -- driver logs left beside the directories nine rounds have removed.  See
# plan_orphanlogs_20260925.py's header for why the directory planner cannot see them and
# why this is NOT the file-level release of round D sec 14.
#
# ORDER: plan_20260925.py (its keep list is an input) -> plan_orphanlogs_20260925.py ->
#        archive_orphanlogs_20260925.py -> this.
# Gates, each with its own exit code -- modelled on retire_files_20260916b.sh:
#   2   no plan list for a tree
#   3   N of M targets already gone (the signature of a stale list; re-plan)
#   4   a target is a symlink or not a regular file
#   5   a target is outside <tree>/work
#   6   the record dir is missing -- run archive_orphanlogs_20260925.py first
#   10  the confirm-time re-plan fails a gate (O5-O8)
#   11  the re-planned lists differ from the plan-time lists (something moved)
#   14  record gate: a target has no manifest row carrying its own size
#   16  the broken-symlink count of the tree rose after the delete
set -u
D="$(cd "$(dirname "$0")" && pwd)"
R=/home/xqian/toolkit-dev/wcp-porting-img
STAMP=20260925
TREES="pdvd pdhd"
CONFIRM=${CONFIRM:-no}
REC=${REC_OVERRIDE:-$R/sbnd/sbnd_xin/archive/records/cleanup-${STAMP}/orphanlogs}

for t in $TREES; do
  [ -s "$D/files_orphanlogs_${t}_${STAMP}.txt" ] \
    || { echo "REFUSING: no plan list for $t -- run plan_orphanlogs_${STAMP}.py"; exit 2; }
done

# pre-count the broken symlinks so gate 16 means something (INTERLOCK 4's rule)
for t in $TREES; do
  find "$R/$t/work" -xtype l 2>/dev/null | wc -l > "$D/prebroken_orphanlogs_${t}_${STAMP}.txt"
done

if [ "$CONFIRM" = yes ]; then
  echo "== re-plan at confirm time"
  PLAN_SUFFIX=confirm python3 "$D/plan_orphanlogs_${STAMP}.py" \
      > "$D/plan_orphanlogs_${STAMP}.confirm.out" 2>&1 \
    || { echo "REFUSING: a gate fails on re-plan ($D/plan_orphanlogs_${STAMP}.confirm.out)"; exit 10; }
  for t in $TREES; do
    cmp -s "$D/files_orphanlogs_${t}_${STAMP}.txt" "$D/files_orphanlogs_${t}_${STAMP}.confirm.txt" \
      || { echo "REFUSING: the $t list moved between plan and confirm -- a live writer, or a work release that has not happened yet"; exit 11; }
  done
  echo "   OK: O5-O8 PASS, lists unchanged"
fi

python3 - "$D" "$STAMP" "$REC" "$TREES" "$CONFIRM" "$R" <<'PY' || exit $?
import os, sys
D, STAMP, REC, TREES, CONFIRM, R = sys.argv[1:7]
rc = 0
for t in TREES.split():
    work  = os.path.realpath(f"{R}/{t}/work")
    files = [l.strip() for l in open(f"{D}/files_orphanlogs_{t}_{STAMP}.txt") if l.strip()]
    gone  = [f for f in files if not os.path.lexists(f)]
    if gone:
        print(f"REFUSING: {len(gone)} of {len(files)} {t} targets already gone -- re-plan"); sys.exit(3)
    bad = [f for f in files if os.path.islink(f) or not os.path.isfile(f)]
    if bad:
        print(f"REFUSING: {len(bad)} {t} targets are not regular files, e.g. {bad[0]}"); sys.exit(4)
    out = [f for f in files
           if os.path.dirname(os.path.realpath(f)) != work
           or not os.path.basename(f).startswith(".batch_pr_")]
    if out:
        print(f"REFUSING: {len(out)} {t} targets are not .batch_pr_* directly in {work}, e.g. {out[0]}"); sys.exit(5)
    man = f"{REC}/{t}-orphanlogs.manifest.tsv"
    if not os.path.exists(man):
        print(f"REFUSING: no record for {t} ({man}) -- run archive_orphanlogs_{STAMP}.py first"); sys.exit(6)
    # record gate: every target must have a manifest row carrying ITS OWN size.  A row that
    # merely names the path cannot tell a freeze of this file from a freeze of an older one.
    rows = {}
    for line in open(man):
        if line.startswith("#"): continue
        p, size, _mt, _d = line.rstrip("\n").split("\t")
        rows[os.path.join(R, p)] = int(size)
    missing = [f for f in files if rows.get(f) != os.path.getsize(f)]
    if missing:
        print(f"REFUSING: {len(missing)} of {len(files)} {t} targets have no manifest row with their own size, e.g. {missing[0]}"); sys.exit(14)
    tot = sum(os.path.getsize(f) for f in files)
    print(f"== {t}: {len(files)} logs, {tot/2**30:.2f} GiB, frozen 1:1 in {os.path.basename(man)}")
    if CONFIRM != "yes":
        print(f"   DRY RUN -- nothing removed.  e.g. {os.path.basename(files[0])}")
        continue
    n = 0
    for f in files:
        try:
            os.unlink(f); n += 1
        except OSError as e:
            print(f"   FAILED {f}: {e}"); rc = 1
    print(f"   removed {n}/{len(files)}")
sys.exit(rc)
PY

if [ "$CONFIRM" = yes ]; then
  for t in $TREES; do
    before=$(cat "$D/prebroken_orphanlogs_${t}_${STAMP}.txt")
    after=$(find "$R/$t/work" -xtype l 2>/dev/null | wc -l)
    echo "== $t broken symlinks: $before -> $after"
    [ "$after" -le "$before" ] || { echo "REFUSING (after the fact): the $t broken-symlink count ROSE"; exit 16; }
  done
fi
echo "done (CONFIRM=$CONFIRM)"

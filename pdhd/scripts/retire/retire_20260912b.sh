#!/bin/bash
# Cleanup round 2026-09-12 (doc 105) -- deletion driver.  DRY RUN unless CONFIRM=yes.
#
#   ./retire_20260912b.sh 1 [tree ...]
#   CONFIRM=yes ./retire_20260912b.sh 1 pdvd
#
# Fork of retire_20260910.sh; the keep test is doc 104's, unchanged (see
# plan_20260912b.py's header).  THE ORDER IS NOT OPTIONAL:
#     1. python3 archive_records_20260912b.py 1     freeze the record layer (M13)
#     2. this driver, CONFIRM=yes                  delete
# No materialise step: the planner's transitive closure pulls any borrowed-from
# dir back into KEEP, so INTERLOCK 2 reads zero.
#
# WHAT CHANGED FROM 09-10, AND THE DEFECT BEHIND IT.  The 09-10 INTERLOCK A
# re-ran the planner at confirm time with no arguments.  The planner writes
# tier1_*, keep_* and prebroken_* for EVERY tree, so `1 pdhd` on 2026-09-11
# 05:00 silently overwrote keep_pdvd_20260910.txt -- a COMMITTED record, +120
# p82vprod lines -- and replaced every prebroken_* baseline with a post-release
# count.  The guard is sound; its side effect was a record overwrite (M13).
# Now: (a) the re-plan runs with PLAN_SUFFIX=confirm, so it writes
# <name>.confirm.txt and the plan-time files are never touched, and (b) it is
# given $TREES, the same variable the deletion loop iterates, so it cannot even
# compute a tree this invocation does not act on.
#
# THE TRAP THIS STILL GUARDS (doc 100 catch 2): refuse unless the tier file
# exists, is non-empty, and EVERY line still exists on disk.  Always compare its
# counts against what plan_20260912b.py printed before typing CONFIRM=yes.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
STAMP=20260912b

TIER=${1:?tier (1)}; shift || true
case "$TIER" in 1) ;; *) echo "this round has one tier: 1"; exit 1;; esac
TREES=${*:-sbnd pdvd pdhd}
CONFIRM=${CONFIRM:-no}

# ---- INTERLOCK A: RE-PLAN AT CONFIRM TIME -------------------------------
# On 09-06 a peer session started 11 minutes into planning and wrote two new
# arms; on 09-04 a live round created SEVEN new families between plan and
# confirm.  A tier file frozen at plan time cannot see them.
#
# SCOPED TO THE TIER *AND* THE TREES BEING RUN.  The 09-08 version scoped to the
# tier only, and on 2026-09-10 that reproduced the very defect its own comment
# describes, on a new axis: this round has ONE tier and THREE trees, so running
# `1 sbnd` and `1 pdvd` first emptied their tier files -- correctly, there is
# nothing left to release there -- and the next invocation, `1 pdhd`, compared
# ALL tier files, saw pdvd go 9355 -> 0, and refused with a peer-session alarm
# caused by the round's own completed passes.  An already-executed tree is not
# a peer.  Compare only the trees this invocation will touch.
# NOTE the 09-06 bug this avoids: ${TIER} must be assigned ABOVE this block,
# and so must TREES.
if [ "${CONFIRM:-no}" = yes ] && [ "${REPLAN:-yes}" = yes ]; then
  echo "== INTERLOCK A: re-planning before deleting (peer-session guard)"
  if ! PLAN_SUFFIX=confirm python3 "$D/plan_${STAMP}.py" $TREES > "$D/plan_${STAMP}.confirm.out" 2>&1; then
    echo "   REFUSING: the plan no longer passes its interlocks. See"
    echo "   $D/plan_${STAMP}.confirm.out"; exit 10
  fi
  changed=0
  for t in $TREES; do
    b="tier${TIER}_${t}_${STAMP}.txt"
    c="tier${TIER}_${t}_${STAMP}.confirm.txt"
    [ -f "$D/$c" ] || { echo "   REFUSING: the re-plan wrote no $c"; exit 12; }
    cmp -s "$D/$b" "$D/$c" || { echo "   CHANGED since plan time: $b vs $c"; changed=1; }
  done
  if [ "$changed" != 0 ]; then
    echo "   REFUSING: the tier files moved between plan and confirm -- that is"
    echo "   what a live peer looks like.  Review the diff, then re-confirm."
    exit 11
  fi
  echo "   OK: all interlocks still PASS; the tier file of every tree being"
  echo "   run ($TREES) is unchanged."
fi

for t in $TREES; do
  TF="$D/tier${TIER}_${t}_${STAMP}.txt"
  echo "=================================================================="
  echo "== $t   tier $TIER   file: $TF"
  if [ ! -s "$TF" ]; then
    echo "   (empty or missing tier file -- nothing planned for this tree/tier)"; continue
  fi
  n=$(wc -l < "$TF"); miss=0; kb=0
  while read -r p; do
    if [ -e "$p" ]; then kb=$((kb + $(du -sk "$p" | cut -f1))); else miss=$((miss+1)); fi
  done < "$TF"
  echo "   lines $n | present $((n-miss)) | already gone $miss | $(echo "scale=2; $kb/1048576" | bc) GiB"
  if [ "$miss" -gt 0 ]; then
    echo "   REFUSING: $miss of $n targets are already gone -- that is the"
    echo "   signature of pointing at a PREVIOUS round's tier list. Re-plan."
    exit 3
  fi
  bad=$(while read -r p; do [ -L "$p" ] && echo "$p"; done < "$TF")
  if [ -n "$bad" ]; then echo "   REFUSING: symlink in the target list:"; echo "$bad"; exit 4; fi
  out=$(grep -cv '^/home/xqian/toolkit-dev/wcp-porting-img/' "$TF" || true)
  if [ "$out" != 0 ]; then echo "   REFUSING: $out targets outside wcp-porting-img"; exit 5; fi
  # The record layer must be frozen BEFORE the bytes go (M13).  The 09-06 driver
  # had two CONFIRM-only bugs here -- a path that never existed, and a gate on
  # the LAST tree so the first two were already deleted when it refused.  This
  # is the corrected shape: real output path, per tree AND per tier, every tree.
  REC="$D/../../../sbnd/sbnd_xin/archive/records/cleanup-${STAMP}/${t}-tier${TIER}"
  if [ "$CONFIRM" = yes ] && [ ! -d "$REC" ]; then
    echo "   REFUSING: $REC missing -- run 'python3 archive_records_${STAMP}.py $TIER' first."
    exit 6
  fi

  if [ "$CONFIRM" = yes ]; then
    echo "   DELETING..."
    xargs -a "$TF" -d '\n' rm -rf
    echo "   done, rc=$?"
  else
    echo "   DRY RUN -- first 3 targets:"; head -3 "$TF" | sed 's/^/      /'
  fi
done

echo
echo "POST-STATE CHECK.  interlock 4 recorded the PRE-existing broken-symlink"
echo "count per tree (prebroken_<tree>_20260912b.txt); this number only means"
echo "something compared against that."
for t in pdhd pdvd sbnd/sbnd_xin; do
  echo "   $t: $(find /home/xqian/toolkit-dev/wcp-porting-img/$t -xtype l 2>/dev/null | wc -l) broken symlinks"
done
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."

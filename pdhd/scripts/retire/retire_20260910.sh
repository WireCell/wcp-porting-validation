#!/bin/bash
# Cleanup round 2026-09-10 (doc 104) -- deletion driver.  DRY RUN unless CONFIRM=yes.
#
#   ./retire_20260910.sh 1 [tree ...]
#   CONFIRM=yes ./retire_20260910.sh 1 pdvd
#
# WHAT MAKES THIS ROUND DIFFERENT, AND WHY IT IS BIGGER THAN ANY BEFORE IT.
# The owner changed the keep test: "we want to keep the latest production, but
# for work* directory that are intermediate results, we can retire them to save
# disk".  Every round from 09-02 to 09-08 kept an arm because a doc cited it and
# so released nothing from pdvd or pdhd at all.  Here a committed doc IS the
# record and the per-event bytes under it are what goes.  An arm survives only
# if it is substrate, the latest production, a hand-scan source, part of a live
# round, or evidence for a decision that is still OPEN.  plan_20260910.py states
# each of those and INTERLOCK 13/14/15 enforce them.
#
# THE ORDER IS NOT OPTIONAL:
#     1. python3 archive_records_20260910.py 1     freeze the record layer (M13)
#     2. this driver, CONFIRM=yes                  delete
# There is no MATERIALISE step this round and that is a measurement, not an
# omission: nothing kept borrows from anything released, because the planner's
# transitive closure pulls any such dir back into KEEP (INTERLOCK 2 then reads
# zero).  The 09-08 round needed one because it deliberately exempted grp0825.
#
# THE TRAP THIS GUARDS (doc 100 catch 2): the 08-31 driver built its tier
# filename by interpolation, a literal rename missed it, and the first dry run
# silently targeted the PREVIOUS round's already-deleted list and reported
# dirs=0.  So this refuses unless the tier file exists, is non-empty, and EVERY
# line still exists on disk.  Always compare its counts against what
# plan_20260910.py printed before typing CONFIRM=yes.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
STAMP=20260910

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
  mkdir -p "$D/.preplan-$STAMP"
  for t in $TREES; do
    cp "$D/tier${TIER}_${t}_${STAMP}.txt" "$D/.preplan-$STAMP/" 2>/dev/null || true
  done
  if ! python3 "$D/plan_${STAMP}.py" > "$D/plan_${STAMP}.confirm.out" 2>&1; then
    echo "   REFUSING: the plan no longer passes its interlocks. See"
    echo "   $D/plan_${STAMP}.confirm.out"; exit 10
  fi
  changed=0
  for t in $TREES; do
    b="tier${TIER}_${t}_${STAMP}.txt"
    cmp -s "$D/$b" "$D/.preplan-$STAMP/$b" || { echo "   CHANGED since plan time: $b"; changed=1; }
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
echo "count per tree (0 / 0 / 0 on 2026-09-08); this number only means"
echo "something compared against that."
for t in pdhd pdvd sbnd/sbnd_xin; do
  echo "   $t: $(find /home/xqian/toolkit-dev/wcp-porting-img/$t -xtype l 2>/dev/null | wc -l) broken symlinks"
done
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."

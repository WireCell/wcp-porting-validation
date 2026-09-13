#!/bin/bash
# ~/tmp sweep, cleanup round 2026-09-12 (doc 105).  DRY RUN unless CONFIRM=yes.
#
#   ./sweep_tmp_20260912.sh <tier>        tier = 1 or 2
#
# STEP 0 IS NOT IN THIS FILE, AND IT COMES FIRST:
#     CONFIRM=yes python3 dedup_pins_20260912.py
# 65 pin roots, 61.45 GiB of pinned binaries -> 30.31 GiB by hardlinking
# byte-identical files (31.14 GiB, measured by the dry run 2026-09-12).  It
# deletes nothing; every pin stays complete and runnable.  The p83..p96 rounds
# each copied a whole ~4.75 GiB pin (cudnn included) and none of it had ever
# been deduplicated.  Run it before tier 2: tier 2 frees UNIQUE bytes, and a
# pin's unique bytes are what is left after dedup.
#
# THE RULE, carried unchanged from doc 100 / 09-05 / 09-06 / 09-10: this sweep
# never removes a *.log/.txt/.md/.json/.tsv/.npy on its own -- a file is a record
# by FUNCTION, not extension.  Tier 1 removes git worktrees whose every commit is
# on the remote (git itself refuses a dirty one); tier 2 removes only pure-.so
# pin directories.
#
# WHAT IS DELIBERATELY NOT HERE (each is an owner decision, stated in doc 105):
#   wt-merge / wt-premerge (~3.5 GiB, the install/ trees left after 09-10 tier 2)
#     mg10's env_{A,B}.sh put them on LD_LIBRARY_PATH and mg10 still has no
#     committed doc; a missing LD_LIBRARY_PATH entry is SILENTLY ignored (M1).
#   mg10 (2.63)            the same undocumented round.
#   claude-25225 (4.66)    session scratchpads, a live peer's and this one's included.
#   h2x / p8x / p9x round dirs' non-.so content (prep sets, gate logs): records.
#   top-level pins d47_libpin, d08_libpin, d09_libpin, d41/42/45_libpin,
#     d97b-libsnap: each backs an arm PROTECTED.txt or this round keeps.
set -u
T=/home/xqian/tmp
R=/home/xqian/toolkit-dev/wcp-porting-img
D="$(cd "$(dirname "$0")" && pwd)"
STAMP=20260912
TIER=${1:?tier (1 or 2)}
CONFIRM=${CONFIRM:-no}
REMOTE=https://github.com/WireCell/wcp-porting-validation.git

newest_age_min() {   # minutes since the newest FILE anywhere under $1 was written
  python3 - "$1" <<'PY'
import os, sys, time
m = 0
for cur, s, fs in os.walk(sys.argv[1]):
    s[:] = [x for x in s if not os.path.islink(os.path.join(cur, x))]
    for f in fs:
        try: m = max(m, os.lstat(os.path.join(cur, f)).st_mtime)
        except OSError: pass
print(int((time.time() - m) / 60) if m else 99999)
PY
}
cwd_inside() {       # any process of ours whose cwd is under $1
  for p in /proc/[0-9]*; do
    c=$(readlink "$p/cwd" 2>/dev/null) || continue
    case "$c/" in "$1"/*) echo "${p#/proc/}"; return 0;; esac
  done
  return 1
}

case "$TIER" in
 1)
  # ---- TIER 1: push worktrees whose commit is ON THE REMOTE -----------------
  # A round pushes by cherry-picking onto FETCH_HEAD in a throwaway worktree
  # (local main is diverged from origin/main; memory "push over https").  Ten of
  # those are still registered: seven ~/tmp/push2x from the doc pdhd/21-23
  # pushes and three inside DEAD sessions' scratchpads (transcripts last written
  # 09-10 06:54, 09-12 06:33, 09-12 14:10; the live peer is 1cab00a8 and has
  # none).  Measured 2026-09-12: every HEAD is an ancestor of the remote main
  # tip ff8f3ee4, so removing them loses no commit.  ~5 GiB.
  #
  # The verb is `git worktree remove`, never rm -rf: rm leaves a stale
  # registration, and git refuses by itself to remove a worktree with modified
  # or untracked files -- a second guard that costs nothing.  The ancestry test
  # is re-derived HERE from a fresh ls-remote, because the local origin/main
  # tracking ref is stale in this repo (doc 102 sec 0).
  S=$T/claude-25225/-home-xqian-toolkit-dev-toolkit
  WTS="$T/push22 $T/push22c $T/push22d $T/push23 $T/push24 $T/push25 $T/push26
       $S/1fce3b9b-3cb7-4171-9cbd-f3e5eedce907/scratchpad/pushwt
       $S/4088b3b5-c570-48a5-8fe7-659d0a9d58a6/scratchpad/pushwt
       $S/d9d688aa-85d3-4ebf-900f-e2f746766f40/scratchpad/pushwt"
  echo "== TIER 1: push worktrees of wcp-porting-img whose HEAD is on the remote"
  tip=$(timeout 60 git ls-remote "$REMOTE" refs/heads/main 2>/dev/null | cut -f1)
  [ -n "$tip" ] || { echo "   REFUSING: could not read the remote main tip"; exit 5; }
  git -C "$R" cat-file -e "$tip" 2>/dev/null || {
    echo "   REFUSING: remote tip $tip is not in the local object store -- fetch first"; exit 6; }
  echo "   remote main tip: $tip"
  REG=$(git -C "$R" worktree list --porcelain | sed -n 's/^worktree //p')
  GO=()
  for w in $WTS; do
    [ -d "$w" ] || { echo "   (absent, skipping) $w"; continue; }
    grep -qxF "$w" <<< "$REG" || { echo "   REFUSING $w: not a registered worktree of $R"; exit 7; }
    h=$(git -C "$w" rev-parse HEAD) || { echo "   REFUSING $w: no HEAD"; exit 7; }
    git -C "$R" merge-base --is-ancestor "$h" "$tip" || {
      echo "   REFUSING $w: HEAD $h is NOT on the remote -- removing it could lose a commit"; exit 8; }
    n=$(git -C "$w" status --porcelain --untracked-files=all | wc -l)
    [ "$n" = 0 ] || { echo "   REFUSING $w: $n modified/untracked path(s)"; exit 9; }
    a=$(newest_age_min "$w")
    [ "$a" -ge 60 ] || { echo "   REFUSING $w: a file was written $a min ago -- live"; exit 10; }
    if pid=$(cwd_inside "$w"); then echo "   REFUSING $w: process $pid has its cwd inside"; exit 11; fi
    echo "   $(du -sh "$w" | cut -f1) | idle ${a} min | HEAD ${h:0:8} on remote | $w"
    GO+=("$w")
  done
  [ "${#GO[@]}" -gt 0 ] || { echo "   nothing to do"; exit 0; }
  if [ "$CONFIRM" = yes ]; then
    for w in "${GO[@]}"; do
      git -C "$R" worktree remove "$w" && echo "   removed $w" || { echo "   git refused $w -- stopping"; exit 12; }
    done
  else
    echo "   would run: git -C $R worktree remove <each of the ${#GO[@]} above>"
  fi
  ;;

 2)
  # ---- TIER 2: nested pins whose round has NO surviving arm ---------------
  # SELF-SEQUENCING, carried from 09-10 tier 3: pins_of_dead_rounds_20260912.py
  # recomputes at RUN time which arm families are on disk, so run this AFTER the
  # work/ release (and after step 0), when it frees the most.  VALUE-FIRST: a
  # pin is kept if any file that names it also names an arm still on disk
  # (libpin_p75 backed p79vprod; d66/libpin backs the smx3/smx4 sources).
  # 2026-09-12: the round grammar grew an `h` (h18..h27), so the script now
  # matches h-rounds' dirs and arm tokens as well.
  echo "== TIER 2: nested pins of rounds with no surviving arm"
  python3 "$D/pins_of_dead_rounds_${STAMP}.py" | sed 's/^/   /'
  mapfile -t T2 < <(python3 "$D/pins_of_dead_rounds_${STAMP}.py" --list)
  [ "${#T2[@]}" -gt 0 ] || { echo "   nothing releasable yet -- run the work/ release first"; exit 0; }
  GO=()
  for d in "${T2[@]}"; do
    [ -e "$d" ] || { echo "   REFUSING: $d is already gone -- which round's list is this?"; exit 3; }
    kb=$(find "$d" -type f ! -name '*.so*' -printf '%k\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
    if [ "${kb:-0}" -ge 2048 ]; then
      echo "   SKIP $d: ${kb} KiB of non-.so content -- not the regenerable class"; continue
    fi
    a=$(newest_age_min "$d")
    [ "$a" -ge 60 ] || { echo "   SKIP $d: written $a min ago -- live"; continue; }
    GO+=("$d")
  done
  [ "${#GO[@]}" -gt 0 ] || { echo "   nothing left after the purity/liveness filter"; exit 0; }
  if [ "$CONFIRM" = yes ]; then rm -rf "${GO[@]}"; echo "   removed ${#GO[@]} pin dir(s)"
  else printf '   would remove: %s\n' "${GO[@]}"; fi
  ;;
 *) echo "this round has tiers 1 and 2 (step 0 is dedup_pins_${STAMP}.py)"; exit 1;;
esac
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."

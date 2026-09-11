#!/bin/bash
# ~/tmp sweep, cleanup round 2026-09-10 (doc 104).  DRY RUN unless CONFIRM=yes.
#
#   ./sweep_tmp_20260910.sh <tier>        tier = 1
#
# READ THIS FIRST: THE DEDUP ALREADY TOOK MOST OF WHAT A SWEEP COULD TAKE.
# dedup_pins_20260910.py ran first and hardlinked 12553 files across 61 pin
# roots -- 59.92 GiB of pinned binaries now occupy 28.68, with nothing deleted
# and every pin still complete and runnable.  So the numbers below are UNIQUE
# bytes (each inode charged once), not `du`, which now over-reports a shared
# pin badly.  Deleting a deduped pin frees only what is unique to it.
#
# THE RULE, carried from doc 100 / 09-05 / 09-06 unchanged: this sweep removes
# only whole lib*/ snapshot dirs.  It never removes a *.log/.txt/.md/.json/
# .tsv/.npy on its own -- a libsnap is regenerable from the commit its doc
# records, a gate log is not, and a file is a record by FUNCTION not extension.
# Every candidate below is >= 99% *.so by bytes, checked at run time by
# pure_so(), so dropping one removes zero record bytes by construction.
#
# WHAT IS DELIBERATELY NOT HERE, and why -- these are the four biggest items in
# ~/tmp and none of them is sweepable:
#   wt-premerge (10.07 GiB) and wt-merge (7.72)  REGISTERED git worktrees of the
#     toolkit (`git -C toolkit worktree list`), both written 2026-09-10 08:49.
#     Removing the directory out from under the repo leaves a stale
#     registration; the correct verb is `git worktree remove`, and that is the
#     owner's call on a merge that landed today (98140fee).
#   claude-25225 (4.17)  live session scratchpads, this session's included.
#   p82 (3.67) and mg10 (2.63)  LIVE ROUNDS.  doc pdvd/82 is uncommitted and
#     was writing arms at 21:38 today.  AGE IS NOT LIVENESS -- derive it from
#     ps, as doc 100 had to learn from an 18 GiB scratchpad that looked idle.
#
# M1 SHAPE, stated once: a missing LD_LIBRARY_PATH directory is SILENTLY
# ignored and falls back to live local/lib, so a swept pin turns an arm from
# re-runnable into merely re-readable with no error anywhere.  Every path below
# was grepped across both repos' doc and script roots first.
set -u
T=/home/xqian/tmp
R=/home/xqian/toolkit-dev/wcp-porting-img
D="$(cd "$(dirname "$0")" && pwd)"
STAMP=20260910
TIER=${1:?tier (1, 2 or 3)}
CONFIRM=${CONFIRM:-no}

run() {
  for p in "$@"; do
    [ -e "$p" ] || { echo "   REFUSING: $p is already gone -- which round's list is this?"; exit 3; }
  done
  if [ "$CONFIRM" = yes ]; then rm -rf "$@"; else echo "   would remove: $*"; fi
}
pure_so() {
  local d=$1
  local kb; kb=$(find "$d" -type f ! -name '*.so*' -printf '%k\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
  [ "${kb:-0}" -lt 2048 ] || { echo "   REFUSING $d: ${kb} KiB of non-.so content -- not the regenerable class"; exit 4; }
}
CITROOTS="$R/pdhd/docs $R/pdhd/scripts $R/pdvd/docs $R/pdvd/scripts
          $R/sbnd/sbnd_xin/docs $R/sbnd/sbnd_xin/scripts
          /home/xqian/toolkit-dev/toolkit/clus/docs"
cites() {   # how many places name this exact basename; excludes THIS round's files
  grep -rlF --exclude-dir=retire --exclude="104_*" "$(basename "$1")" $CITROOTS 2>/dev/null | wc -l
}
uniq_gib() { python3 - "$1" <<'PY'
import os,sys
seen=set(); t=0
for cur,s,fs in os.walk(sys.argv[1]):
    for f in fs:
        try: st=os.lstat(os.path.join(cur,f))
        except OSError: continue
        if st.st_ino in seen: continue
        seen.add(st.st_ino); t+=st.st_size
print(f"{t/2**30:.2f}")
PY
}

# ---- TIER 1: pure-.so pins of CLOSED rounds -------------------------------
# Every one of these belongs to a round whose doc is committed and whose arms
# this round releases, so the pin outlives nothing.  The three pins named in a
# PROTECTED.txt (d47_libpin/new11, pdhdstm_libpin, and the doc-81 libpin_p81*)
# are NOT here; neither is libpin_p75, which is the binary behind the kept
# production arm p79vprod.
T1="$T/d46_libpin $T/d30_libpin $T/d30_libpin_post $T/d30_libpin_r3
    $T/d30_libpin_r3g $T/d30r2_libpin"
# A PIN GOES WITH ITS ARMS, and this tier is what is LEFT after applying that.
# Derived, not judged: for each candidate pin, which arm families of its round
# survive in this round's own keep_*.txt?  Fifteen candidates were pulled back
# by that test and the reasons are worth reading, because four of them look
# releasable by name:
#   d08_libpin   d08cap10 / d08goff / d08pv30on / d08pv30off SURVIVE -- they are
#                hand-scan sources, which is exactly what the owner asked to keep
#   d09_libpin   d09 / d09ctl / d09ctl2 are pdhd SUBSTRATE (2010 inbound links)
#   d45/d41/d42/d44_libpin   d45prod, d41prod/d41prov, d42fit, d44prod/ref/sig*
#                are all still named in PROTECTED.txt as shipped-value evidence
#   d102m-libsnap  the LATEST PRODUCTION binary (work-*-d102m / d102mpr)
#   d145_libpin*   pr/148 is OPEN and reads work-*-d145np; d146_libpin* back
#                  work-*-d146sv25.  Both sets survive
#   d144_libpin*   AMBIGUOUS and therefore kept: work-*-d144fixprod went in the
#                  09-08 round, but the sentinel suite's work-s144pos/neg-* layer
#                  survives and the `s144` token does not contain `d144`, so the
#                  substring derivation cannot clear it.  Ambiguity keeps a pin.
# What is left is doc pdvd/30's PR-memory family and doc 46, whose arms this
# round releases in full.
# DROPPED FROM THE TIER BY pure_so(), and kept as a result -- d58_libpin,
# xtrack_libpin, pdhd02_libpin and d97b-libsnap each carry 13448 non-.so files
# (~275 MiB): they are FULL local/lib snapshots, not the ~19-object pin class,
# and the round that just ended proved what a partial pin costs (doc pdvd/81
# sec 5.3: an old clus lib against a rebuilt root lib spun one job for 85
# minutes at 867 MB with no error line).  d97b-libsnap is additionally named
# in sbnd_xin/scripts/retire/PROTECTED.txt as the production binary.  The
# guard found all four; none was removed by hand.

case "$TIER" in
 1)
  echo "== TIER 1: pure-.so pins of closed rounds"
  tot=0
  for d in $T1; do
    if [ ! -e "$d" ]; then echo "   (absent, skipping) $d"; continue; fi
    pure_so "$d"
    c=$(cites "$d"); g=$(uniq_gib "$d")
    echo "   $g GiB unique | cited by $c file(s) | $(basename "$d")"
    if [ "$c" -gt 0 ]; then
      echo "     NOTE: named somewhere -- a pin is regenerable from the commit its"
      echo "     doc records, so a citation does not block it, but read the line first:"
      grep -rlF --exclude-dir=retire --exclude="104_*" "$(basename "$d")" $CITROOTS 2>/dev/null | head -2 | sed 's/^/       /'
    fi
  done
  echo
  echo "   TIER 1 total unique: $(python3 - $T1 <<'PY'
import os,sys
seen=set(); t=0
for root in sys.argv[1:]:
    for cur,s,fs in os.walk(root):
        for f in fs:
            try: st=os.lstat(os.path.join(cur,f))
            except OSError: continue
            if st.st_ino in seen: continue
            seen.add(st.st_ino); t+=st.st_size
print(f"{t/2**30:.2f} GiB")
PY
)"
  run $T1
  ;;

 2)
  # ---- TIER 2: the BUILD trees of the two merge worktrees -----------------
  # 14.5 GiB of compile output, and the honest unit here is build/ -- NOT the
  # worktree.  `git worktree remove` would take install/ with it, and
  # ~/tmp/mg10/env_{A,B}.sh put THAT on LD_LIBRARY_PATH:
  #     env_A.sh -> /home/xqian/tmp/wt-premerge/install/lib   (arm A, 8b1374f9)
  #     env_B.sh -> /home/xqian/tmp/wt-merge/install/lib      (arm B, adf0fd9c)
  # A missing LD_LIBRARY_PATH directory is SILENTLY ignored and falls back to
  # the shared /home/xqian/toolkit-dev/local/lib, so removing the worktrees
  # would not fail -- it would make both merge-gate arms quietly re-run against
  # the WRONG libraries.  That is the M1 shape and doc 100 already paid for it
  # once, when a ~/tmp sweep deleted the pin a runner named.
  #
  # MEASURED 2026-09-10: every reference anywhere in both repos and in
  # ~/tmp/mg10 is to install/, cfg/ or the worktree root.  NOTHING names
  # build/.  install/ is self-contained -- 0 symlinks, 0 hardlinks into build,
  # and both binaries run and report the exact commits RESULTS.md names.
  # The merge itself is safe either way: 98140fee and 8b1374f9 are both
  # reachable from apply-pointcloud, so no commit lives only in a worktree.
  echo "== TIER 2: build/ trees of the merge worktrees (worktrees themselves stay)"
  T2=""
  for w in $T/wt-merge $T/wt-premerge; do
    [ -d "$w" ] || { echo "   (absent, skipping) $w"; continue; }
    # (a) nothing may be lost: HEAD must be reachable from a branch
    h=$(git -C "$w" rev-parse HEAD 2>/dev/null) || { echo "   REFUSING $w: not a worktree"; exit 5; }
    git -C "$w" branch -a --contains "$h" 2>/dev/null | grep -q . || {
      echo "   REFUSING $w: HEAD $h is on no branch -- removing anything here loses a commit"; exit 6; }
    # (b) no uncommitted tracked work
    d=$(git -C "$w" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
    [ "$d" = 0 ] || { echo "   REFUSING $w: $d modified tracked file(s)"; exit 7; }
    # (c) the arm must survive: install/ present, self-contained, and RUNNING
    [ -x "$w/install/bin/wire-cell" ] || { echo "   REFUSING $w: no install/bin/wire-cell to fall back on"; exit 8; }
    n=$(find "$w/install" -type l -printf '%l
' 2>/dev/null | grep -c build)
    [ "$n" = 0 ] || { echo "   REFUSING $w: install/ has $n symlink(s) into build/"; exit 9; }
    v=$(LD_LIBRARY_PATH=$w/install/lib $w/install/bin/wire-cell --version 2>&1 | head -1)
    case "$v" in HEAD-*) ;; *) echo "   REFUSING $w: install binary does not run ($v)"; exit 10;; esac
    # (d) nothing may name build/
    c=$(grep -rhoE "$w/build[A-Za-z0-9/._-]*" $T/mg10 $R 2>/dev/null | wc -l)
    [ "$c" = 0 ] || { echo "   REFUSING $w: $c reference(s) to $w/build"; exit 11; }
    [ -d "$w/build" ] || { echo "   (build/ already gone) $w"; continue; }
    echo "   $(du -sh --apparent-size $w/build | cut -f1) | $w/build  (install/ runs: $v)"
    T2="$T2 $w/build"
  done
  [ -n "$T2" ] || { echo "   nothing to do"; exit 0; }
  run $T2
  echo "   NOTE: the worktrees and their install/ trees are KEPT.  When the owner"
  echo "   is done with the merge arms, 'git -C <wt> ...' then"
  echo "   'git -C /home/xqian/toolkit-dev/toolkit worktree remove <wt>' releases"
  echo "   the remaining 3.1 GiB -- and env_{A,B}.sh must be retired with them."
  ;;

 3)
  # ---- TIER 3: nested pins whose round has NO surviving arm ---------------
  # SELF-SEQUENCING.  pins_of_dead_rounds_20260910.py recomputes which arm
  # families are on disk at RUN time, so running this before the pdhd release
  # correctly frees less, and running it after frees more.  Nothing is frozen
  # to a moment, which is the opposite of a tier file.
  #
  # The test is VALUE-FIRST.  By NAME, ~/tmp/p75/libpin_p75 belongs to round
  # p75 whose arms are all released -- but d81_arms.sh names that pin and it is
  # the binary behind p79vprod, CURRENT PDVD PRODUCTION (doc pdvd/81 sec 6).
  # ~/tmp/d66/libpin is the same shape: d68_arms.sh ties it to d68a3/d68d4, the
  # smx3/smx4 HAND-SCAN sources.  Both are held.  This is the d08cap10 mistake
  # and the reason the script greps for the pin, pulls the arm tokens out of
  # every file that names it, and keeps the pin if ANY of them still exists.
  # It also scans the rounds' OWN runners in ~/tmp: without those, a pin named
  # only by ~/tmp/d62/run_arms.sh reads as "named by 0 files" and looks
  # releasable on no evidence at all.  Adding them took the tier from 7 pins to
  # 2 -- over-keeping is the safe direction.
  echo "== TIER 3: nested pins of rounds with no surviving arm"
  python3 "$D/pins_of_dead_rounds_${STAMP}.py" | sed 's/^/   /'
  mapfile -t T3 < <(python3 "$D/pins_of_dead_rounds_${STAMP}.py" --list)
  [ "${#T3[@]}" -gt 0 ] || { echo "   nothing releasable yet -- run the pdhd release first"; exit 0; }
  # pure_so() EXITS on an impure dir, which is right for a hand-written tier
  # but wrong for a derived one: a single full local/lib snapshot in the list
  # would abort the whole tier.  Here it filters instead, and says so --
  # ~/tmp/d57/pin_lib is one of those (13448 non-.so files, a whole install
  # tree rather than the ~19-object pin class).
  KEEPLIST=()
  for d in "${T3[@]}"; do
    kb=$(find "$d" -type f ! -name '*.so*' -printf '%k\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
    if [ "${kb:-0}" -lt 2048 ]; then KEEPLIST+=("$d")
    else echo "   SKIP $d: ${kb} KiB of non-.so content -- not the regenerable class"; fi
  done
  [ "${#KEEPLIST[@]}" -gt 0 ] || { echo "   nothing left after the purity filter"; exit 0; }
  run "${KEEPLIST[@]}"
  ;;
 *) echo "this round has tiers 1, 2 and 3"; exit 1;;
esac
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."

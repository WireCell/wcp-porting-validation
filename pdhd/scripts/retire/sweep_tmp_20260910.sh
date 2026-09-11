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
TIER=${1:?tier (1)}
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
 *) echo "this round has one tier: 1"; exit 1;;
esac
[ "$CONFIRM" = yes ] || echo -e "\nDRY RUN.  Re-run with CONFIRM=yes to execute."

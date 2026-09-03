#!/bin/bash
# doc 98 round 2 -- release dead Claude session scratchpads from ~/tmp.
#
#   ./scripts/retire/sweep_scratch_20260902.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/sweep_scratch_20260902.sh  # actually delete
#
# ~/tmp/claude-25225/-home-xqian-toolkit-dev-toolkit holds one dir per Claude
# session that has ever run in this tree -- 32 of them, 3.4 GiB, almost all
# from sessions that ended days ago.  They are pure scratch by construction
# (the session-scratchpad contract), and nothing outside a session reads them.
#
# THE ONE HAZARD, and why this is not a bare `find -mtime`.  A PEER SESSION IS
# LIVE in this tree (PDVD doc 25) and it has a scratchpad here too.  Deleting
# it would pull the floor out from under a running session.  There is no way to
# map a scratchpad dir to a live pid, so the rule is conservative and stated:
# KEEP everything modified today, whoever owns it.  That kept 4 dirs on
# 2026-09-02 (this session plus three others) and released 28.
#
# Interlock 2 additionally re-samples mtimes across a 15 s window and refuses if
# any candidate moves -- the same causal liveness test doc 98's main round used,
# for the same reason: age is not liveness, it is only a cheap prior.
#
# BUG FIXED 2026-09-02, AFTER IT FIRED ON THIS SESSION.  The first version read
# each candidate's TOP-LEVEL DIRECTORY mtime.  A live session writes into
# SUBdirectories (tasks/), which does not touch the parent's mtime, so a running
# session can present an arbitrarily old top-level mtime.  The first run deleted
# this very session's scratchpad, which the harness then recreated.
#
# That was the harmless instance.  The dangerous one was sitting right next to
# it: peer scratchpad b48747db had a top-level mtime of 10:18 while its newest
# file inside was 16:51 -- the live PDVD doc-25 session.  It survived only
# because 10:18 happened to fall on the same calendar day.  The same sweep run
# the next morning would have deleted a running peer's scratchpad.
#
# Both selection and interlock 2 now use the NEWEST mtime ANYWHERE IN THE TREE.
# Never judge a directory's activity by the directory's own mtime.
set -u
SCRATCH=$HOME/tmp/claude-25225/-home-xqian-toolkit-dev-toolkit
CONFIRM=${CONFIRM:-no}
TODAY=$(date +%Y-%m-%d)

[ -d "$SCRATCH" ] || { echo "no scratchpad root at $SCRATCH"; exit 0; }
cd "$SCRATCH" || exit 2

CANDS=""
for d in */; do
  d=${d%/}
  [ -d "$d" ] || continue
  # newest mtime anywhere in the tree -- NOT the dir's own (see header)
  if [ -z "$(find "$d" -newermt "$TODAY 00:00" -print -quit 2>/dev/null)" ]; then
    CANDS="$CANDS $d"
  fi
done
n=$(echo $CANDS | wc -w)
[ "$n" -gt 0 ] || { echo "nothing older than today; nothing to do"; exit 0; }

echo "== interlock 1: keep everything modified today (a live peer session owns one) =="
for d in */; do
  d=${d%/}
  newest=$(find "$d" -newermt "$TODAY 00:00" -print -quit 2>/dev/null)
  if [ -n "$newest" ]; then
    echo "  KEEP $d  (dir mtime $(date -r "$d" '+%m-%d %H:%M'), but active today inside)"
  fi
done

echo "== interlock 2: no candidate may move over a 15s window =="
snap() { for d in $CANDS; do
           echo "$d:$(find "$d" -printf '%T@\n' 2>/dev/null | sort -rn | head -1)"
         done; }
before=$(snap)
sleep 15
after=$(snap)
if [ "$before" != "$after" ]; then
  echo "REFUSE: a candidate scratchpad changed under us"; diff <(echo "$before") <(echo "$after"); exit 2
fi
echo "  ok -- all $n candidates quiet"

echo
echo "== sweep set: $n dead session scratchpads =="
du -sch $CANDS 2>/dev/null | tail -1

if [ "$CONFIRM" != "yes" ]; then
  echo
  echo "DRY RUN -- nothing deleted."
  echo "To execute:  CONFIRM=yes ./scripts/retire/sweep_scratch_20260902.sh"
  exit 0
fi

echo "== DELETING =="
for d in $CANDS; do
  if [ -n "$(find "$d" -newermt "$TODAY 00:00" -print -quit 2>/dev/null)" ]; then
    echo "REFUSE: $d has activity today somewhere inside"; exit 2
  fi
  rm -rf -- "$d"
done
echo "  removed $n scratchpads"
# this session's own stray interlock-test dirs
for t in "$HOME"/tmp/ilk4-*; do [ -e "$t" ] && rm -rf -- "$t" && echo "  removed $t"; done
echo
du -sh "$HOME/tmp"

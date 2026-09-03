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
  if [ "$(date -r "$d" +%Y-%m-%d)" != "$TODAY" ]; then CANDS="$CANDS $d"; fi
done
n=$(echo $CANDS | wc -w)
[ "$n" -gt 0 ] || { echo "nothing older than today; nothing to do"; exit 0; }

echo "== interlock 1: keep everything modified today (a live peer session owns one) =="
for d in */; do
  d=${d%/}
  [ "$(date -r "$d" +%Y-%m-%d)" = "$TODAY" ] && echo "  KEEP $(date -r "$d" '+%m-%d %H:%M')  $d"
done

echo "== interlock 2: no candidate may move over a 15s window =="
before=$(for d in $CANDS; do echo "$d:$(date -r "$d" +%s)"; done)
sleep 15
after=$(for d in $CANDS; do echo "$d:$(date -r "$d" +%s)"; done)
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
  if [ "$(date -r "$d" +%Y-%m-%d)" = "$TODAY" ]; then echo "REFUSE: $d touched today"; exit 2; fi
  rm -rf -- "$d"
done
echo "  removed $n scratchpads"
# this session's own stray interlock-test dirs
for t in "$HOME"/tmp/ilk4-*; do [ -e "$t" ] && rm -rf -- "$t" && echo "  removed $t"; done
echo
du -sh "$HOME/tmp"

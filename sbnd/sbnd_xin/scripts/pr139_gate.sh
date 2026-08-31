#!/bin/bash
# doc pr/139 phase 1 -- the knob-off byte-identity gate.
#
#   ./scripts/pr139_gate.sh <off-arm-tag> [baseline-arm-prefix]
#
# Baseline defaults to work-pr138r3-flipchk (the post-flip production config with
# NO env, which pr/138 proved byte-identical to the validated work-pr138r2-c90on).
# The pr/139 knobs all default to the shipped behaviour, so the new binary must
# reproduce it exactly.  Exit code is the verdict, never judged through a pipe (M14).
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
BASE=${2:-work-pr138r3-flipchk}
for s in mcp1k mcp2k ncpi0 nuecc48; do
  python3 scripts/pr85_hash_gate.py $BASE-$s ${PR139_ARM:-work-pr139r1}-$TAG-$s \
      > /home/xqian/tmp/pr139-gate-$TAG-$s.log 2>&1
  rc=$?
  echo "gate $TAG $s rc=$rc :: $(grep -E 'events in A|compared archives|missing|unpaired' /home/xqian/tmp/pr139-gate-$TAG-$s.log | tr '\n' ' ') :: $(tail -1 /home/xqian/tmp/pr139-gate-$TAG-$s.log)"
done
echo "GATE $TAG DONE"

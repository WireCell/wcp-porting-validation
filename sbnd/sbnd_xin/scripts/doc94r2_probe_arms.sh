#!/bin/bash
# doc 94 ROUND 2 -- PROBE the entry-rise feature over the full 3067-event
# data sample.  stm_entry_min_cm=1000 is above the feature's range: the DEBUG
# line prints shoulder / shoulder_nofirst / excess for every STM-evaluated
# bundle and NO verdict can move, so this arm doubles as a byte-identity check
# against work-*-d94hadron (the round-2 baseline -- vertex_hadron_guard is
# production as of ref/prod-2026-09-02).
# Full output (PR_EXTRA_STAGES=pr_display), matching the baseline: doc 87 sec
# 5.3 -- a validation arm must never adopt the production minimal line.
set -u
cd -P "$(dirname "$0")/.." || exit 1
export PR_JOBS=${PR_JOBS:-24}
export PR_EXTRA_STAGES=pr_display
rc_all=0
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    ./scripts/doc94r2_arm.sh "work-$s-grp0825" "work-$s-r2probe" data 1000.0 || rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all

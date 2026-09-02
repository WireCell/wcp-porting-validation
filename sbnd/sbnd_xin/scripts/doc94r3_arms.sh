#!/bin/bash
# doc 94 ROUND 3 -- one pass over the full 3067-event data sample.
#   Usage: [PR_JOBS=14] doc94r3_arms.sh <out-suffix> <min_cm> [extra TLA ...]
# min_cm=1000  -> the inert PROBE arm (measures shoulder + kink, moves nothing)
# min_cm=5.0   -> the guard ON at the round-3 operating point
# Baseline for both is work-*-d94hadron (vertex_hadron_guard is production).
set -u
cd -P "$(dirname "$0")/.." || exit 1
SUF=$1; MINCM=$2; shift 2
export PR_JOBS=${PR_JOBS:-14}
export PR_EXTRA_STAGES=pr_display
rc_all=0
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    ./scripts/doc94r3_arm.sh "work-$s-grp0825" "work-$s-$SUF" data "$MINCM" "$@" || rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all

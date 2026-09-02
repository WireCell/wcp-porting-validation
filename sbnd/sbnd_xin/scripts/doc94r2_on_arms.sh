#!/bin/bash
# doc 94 ROUND 2 -- the ENTRY-RISE guard ON over the full 3067-event data
# sample, at the operating point argument (default 5.0 cm).
#   Usage: [PR_JOBS=24] doc94r2_on_arms.sh <out-suffix> <min_cm> [extra TLA ...]
# Baseline for the A/B is work-*-d94hadron: vertex_hadron_guard is SBND
# production as of ref/prod-2026-09-02 and is ON by default in both arms, so
# the difference is this guard alone.
set -u
cd -P "$(dirname "$0")/.." || exit 1
SUF=${1:-r2entry}; MINCM=${2:-5.0}; shift 2 || true
export PR_JOBS=${PR_JOBS:-24}
export PR_EXTRA_STAGES=pr_display
rc_all=0
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    ./scripts/doc94r2_arm.sh "work-$s-grp0825" "work-$s-$SUF" data "$MINCM" "$@" || rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all

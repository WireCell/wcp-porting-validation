#!/bin/bash
# doc 91 sec 12 -- the knob-OFF side of the sentinel re-baseline.
#
# WHY.  The owner scanned the six FAILing sentinels and judged the production
# batch GOOD, so the registry is stale and must be re-baselined against
# work-*-prod0901b.  A threshold is only a guard if it sits BETWEEN the
# fix-alive and fix-dead values; re-measuring only the alive side would produce
# a suite that is green and cannot fail.
#
# BOTH SIDES MUST BE AT THE SAME OPERATING POINT.  The existing negative
# controls (work-sent130neg*) were run 2026-08-29, before the 0.86 EM scale
# flip, doc 77 r3/r4's 17 retired TLAs and the master merge -- exactly the drift
# that made the registry stale in the first place.  Reusing them would rebuild
# the same fault.  So the fix-dead side is re-measured HERE, at the pinned
# prod0901b point, one arm per knob so a difference is attributable.
#
# Binary and cfg are pinned to what prod0901b actually ran, so the ONLY
# difference between each arm and production is the one knob.
set -u
cd "$(dirname "$0")/.." || exit 2
export LD_LIBRARY_PATH=$HOME/tmp/prod0901b-libsnap:${LD_LIBRARY_PATH:-}
export PR_CFG_TREE=$HOME/tmp/prod0901b-cfgsnap
export PR_EXTRA_STAGES=pr_display
export PR_JOBS=${PR_JOBS:-4}
LOGD=$HOME/tmp/doc91-neg; mkdir -p "$LOGD"

# knob-env | production value | out-arm suffix | sample | event
SPECS='
SBND_STEM_BACKFILL_BACK_GUARD=0|true|backguard|mcp2k|47212
SBND_SCCC_BRIDGE_BODY=0|true|sccc|nuecc48|137238
SBND_SHOWER_PASS3_CONE_GUARD_LEN=0|15|p3cone|mcp2k|173819
SBND_STEM_BACKFILL_BACK_DVTX=0|45|backdvtx|mcp1k|292643
SBND_KINE_GF_IMPACT=0|on|gfimpact|mcp2k|393505
SBND_SHOWER_PASS4_PRUNE_GAP2=0|25|prune2|mcp2k|406125
'
rc_all=0
for spec in $SPECS; do
    IFS='|' read -r knob prod tag sample evt <<< "$spec"
    arm="work-91neg-$tag-$sample"
    if [ -e "$arm" ]; then
        echo "SKIP $arm -- exists already (M13: never write into an existing label)"
        continue
    fi
    echo "=== $arm : $knob (production $prod), evt $evt"
    env "$knob" ./run_pr_chain_batch.sh "work-$sample-grp0825" "$arm" data "$evt" \
        > "$LOGD/$arm.log" 2>&1
    rc=$?
    echo "    rc=$rc"
    [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all

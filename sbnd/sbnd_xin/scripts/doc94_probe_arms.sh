#!/bin/bash
# doc 94 -- PROBE the descent feature over the full 3067-event data sample.
# cos_y=1.01 is above the feature's range: the DEBUG line prints cos_y for
# every STM-evaluated bundle and NO verdict can move, so this arm doubles as a
# byte-identity check against work-*-prod0901b.
# Full output (PR_EXTRA_STAGES=pr_display), matching prod0901b: doc 87 sec 5.3
# -- a validation arm must never adopt the production minimal line.
set -u
cd -P "$(dirname "$0")/.." || exit 1
export PR_JOBS=${PR_JOBS:-6}
export PR_EXTRA_STAGES=pr_display
rc_all=0
for s in ncpi0 nuecc48 mcp1k mcp2k; do
    ./scripts/doc94_arm.sh "work-$s-grp0825" "work-$s-d94probe" data 1.01 || rc_all=1
done
echo "ALL DONE rc_all=$rc_all $(date -Is)"
exit $rc_all

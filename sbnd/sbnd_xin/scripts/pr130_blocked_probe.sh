#!/bin/bash
# doc pr/130 item 4 part 8 -- the contention probe.
#
# WCT_SHOWER_BLOCKED_DEBUG tapes every segment a shower's flood-fill REACHED
# and was turned away from because used_segments already held it.  That is the
# one fact the existing ABSORB tape cannot express, and it separates the two
# hypotheses the owner's scan left open:
#
#   BLOCKED line exists for (target shower, missing segment)
#       -> genuine contention; the outcome is order-dependent
#   no BLOCKED line even though the segment is free of that shower
#       -> a reach failure; ordering is irrelevant
#
#   ./scripts/pr130_blocked_probe.sh gate   # env UNSET, for the byte-identical gate
#   ./scripts/pr130_blocked_probe.sh probe  # env SET, emits the tape
set -u
MODE=${1:-probe}
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-6}
export WCT_SHOWER_ABSORB_DEBUG=1
if [ "$MODE" = xclus ]; then
    # doc pr/130 part 10: the cross-cluster rejection tape
    export WCT_SHOWER_XCLUS_DEBUG=1
    TAG=xcon
    declare -A EV
    EV[nuecc48]="122660 342199 469665"
    EV[ncpi0]="463565 105946 21073"
    EV[mcp2k]="181050"
elif [ "$MODE" = xgate ]; then
    TAG=xgate
    declare -A EV
    EV[nuecc48]="122660"
    EV[mcp2k]="181050"
elif [ "$MODE" = probe ]; then
    export WCT_SHOWER_BLOCKED_DEBUG=1
    TAG=blkon
    declare -A EV
    # the six the owner scanned, plus 463565 which he ruled verbally
    EV[nuecc48]="122660 342199 469665"
    EV[ncpi0]="463565 105946 21073"
    EV[mcp2k]="181050"
else
    TAG=blkgate
    declare -A EV
    EV[nuecc48]="122660"
    EV[mcp2k]="181050"
fi
for s in "${!EV[@]}"; do
    ./run_pr_chain_batch.sh work-$s-grp0825 work-pr130r1-$TAG-$s data ${EV[$s]} \
        > /home/xqian/tmp/pr130_${TAG}_$s.log 2>&1
    echo "arm $TAG-$s rc=$?"
done

#!/bin/bash
# doc pr/130 item 1b -- ON arm for the two guard seats, on exactly the events
# the 239-event census says they can touch (scripts/pr130_launder_scan.py).
#
#   shower_pass4_prox_guard_len     = 50 cm  (pr/123's own measured operating
#       point at the sibling pass4_angle seat; the 4 census cases are 110.3,
#       84.0, 76.1 and 51.4 cm, so 50 catches all four)
#   shower_pass3_backfill_guard_len = 15 cm  (pr/124's shipped value at the
#       sibling pass3_cone seat; the census cases include 16.6 and 17.6 cm)
#
# Using the siblings' own thresholds is the point of the round: the same
# predicate should apply at every seat, not a newly fitted one.
#
# Census-affected events, by sample:
#   mcp2k   100222 (prox) 176502 (both) 396222 (backfill) 415278 (backfill)
#   mcp1k   175896 (backfill)
#   nuecc48 137238 (backfill)
set -u
TAG=${1:-gs1}
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR_EXTRA_STAGES=pr_display PR_JOBS=${PR_JOBS:-12}
export WCT_SHOWER_ABSORB_DEBUG=1
export SBND_SHOWER_PASS4_PROX_GUARD_LEN=50
export SBND_SHOWER_PASS3_BACKFILL_GUARD_LEN=15
declare -A EV
EV[mcp2k]="100222 176502 396222 415278"
EV[mcp1k]="175896"
EV[nuecc48]="137238"
for s in mcp2k mcp1k nuecc48; do
  ./run_pr_chain_batch.sh work-$s-grp0825 work-pr130-$TAG-$s data ${EV[$s]} \
      > /home/xqian/tmp/pr130_${TAG}_$s.log 2>&1
  echo "arm $TAG-$s rc=$?"
done

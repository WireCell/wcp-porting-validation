#!/bin/bash
# doc pr/139 sec 8 -- the three arms items 1, 3 and 4 need.  Each knob ALONE on
# the flipped production config (doc pr/138 B4: a gate answering two questions
# answers neither).  Baseline for all three: work-pr139r3-flipchk-*.
#
#   on     item 1  skip_shared + max_impact=30   -- the operating point sec 6.3
#                  supports and sec 8 pre-registers
#   onrh15 item 3  rehome at a 15 cm gap         -- re-priced on the FLIPPED
#                  config (work-pr139r2-onrh15 was measured pre-flip, with
#                  em_start OFF, and em_start changes which segment roots the
#                  daughter the re-home is looking for a host for)
#   onk3   item 4  max_parts = 3                 -- sec 6.1: the k>=3 boundary
#                  mean of 0.800 is max_parts=2 refusing the third cut, not the
#                  kernel erring.  max_seeds is hardcoded 4, so 3 is the cheap arm.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR140_ARM=work-pr140r1
( export SBND_SHOWER_SPLIT_SKIP_SHARED=1 SBND_SHOWER_SPLIT_MAX_IMPACT=30 \
         WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr140_arms.sh on )
( export SBND_SHOWER_SPLIT_REHOME=1 SBND_SHOWER_SPLIT_REHOME_GAP=15 \
         WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr140_arms.sh onrh15 )
( export SBND_SHOWER_SPLIT_PARTS=3 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr140_arms.sh onk3 )
echo "ALL PR140 ARMS DONE"

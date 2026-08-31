#!/bin/bash
# doc pr/139 phase 1 round 2 -- the fixed binary (pin-pr139b) and two questions
# the single-knob arms raised:
#   1  work-pr139r2-off-*     the SAME knob-off config, re-run so the gate label
#                             attaches to the SHIPPED binary, and so a second
#                             gate (r2-off vs r1-off) proves the two builds are
#                             bit-identical and every r1 arm stays valid.
#   2  work-pr139r2-oncomb-*  P1.1 + P1.2 + P1.3 together, the combination the
#                             owner would actually flip.
#   3  work-pr139r2-onrh15-*  P1.4 at a 15 cm gap: at 4 cm it re-homed 6 of 51
#                             and moved no instrument, so the dial gets ONE
#                             priced arm rather than a search.
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR139_ARM=work-pr139r2
( ./scripts/pr139_arms.sh off )
( export SBND_SHOWER_SPLIT_SKIP_SHARED=1 SBND_SHOWER_SPLIT_MAX_IMPACT=12 \
         SBND_SHOWER_SPLIT_EM_START=1 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr139_arms.sh oncomb )
( export SBND_SHOWER_SPLIT_REHOME=1 SBND_SHOWER_SPLIT_REHOME_GAP=15 WCT_SHOWER_SPLIT_DEBUG=1
  ./scripts/pr139_arms.sh onrh15 )
echo "R2 ARMS DONE"

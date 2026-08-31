#!/bin/bash
# doc pr/139 sec 15-16 -- round 2 arms.  FORK of pr140_arms.sh (M10).  Two
# things differ and both matter:
#   * WCT_SHOWER_CONTENT_DEBUG=1 on EVERY arm.  sec 15.1: it was never set on
#     any pr/138 or pr/139 arm, so prep_em_scan wrote zero sidecars and all
#     completeness scoring fell back to the lossy dump join.  Shared membership
#     is the phenomenon items 1 and 2 are about, so it has to be on.
#   * the binary is /home/xqian/tmp/pin-pr140r2 (md5 5d176a30...), which carries
#     shower_split_shed_shared.  The pr139 pin fbff08ec... does not.
#
# Arms, each knob ALONE on the flipped production config:
#   off     nothing set  -- the knob-off GATE vs work-pr139r3-flipchk, AND the
#                           sidecar-based baseline every restatement needs
#   on      skip_shared + max_impact=30           (P1.6, restated on sidecars)
#   coown   skip_shared + max_impact=30 + shed    (sec 15, the one-knob delta vs `on`)
#   onrh15  rehome gap 15                         (sec 12, restated)
#   onk3    max_parts=3                           (sec 13, restated)
set -u
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
export PR140_ARM=work-pr140r2
export PR140_PIN=/home/xqian/tmp/pin-pr140r2
run () { ( export WCT_SHOWER_CONTENT_DEBUG=1 WCT_SHOWER_SPLIT_DEBUG=1 "${@:2}"
           ./scripts/pr140_arms.sh "$1" ) }
run off
run on      SBND_SHOWER_SPLIT_SKIP_SHARED=1 SBND_SHOWER_SPLIT_MAX_IMPACT=30
run coown   SBND_SHOWER_SPLIT_SKIP_SHARED=1 SBND_SHOWER_SPLIT_MAX_IMPACT=30 \
            SBND_SHOWER_SPLIT_SHED_SHARED=1
run onrh15  SBND_SHOWER_SPLIT_REHOME=1 SBND_SHOWER_SPLIT_REHOME_GAP=15
run onk3    SBND_SHOWER_SPLIT_PARTS=3
echo "ALL PR140r2 ARMS DONE"

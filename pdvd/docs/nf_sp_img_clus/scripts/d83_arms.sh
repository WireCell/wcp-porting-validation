#!/usr/bin/env bash
# doc pdvd/83 -- the arms for doc 78 action item 4 (the attached arm near the stop).
#
# Fork by duplication (CLAUDE.md M10) of d82_arms.sh.  The ONLY differences are
# the arm names, the TLAs and the pin.
#
# The pin is a FULL local/lib snapshot (the 572 top-level *.so*, dereferenced),
# not a Clus-only one: doc pdvd/80's wave 1 died with rc=139 at the ROOT-writing
# stage because a Clus-only pin was mixed with a rebuilt libWireCellRoot.
#
# Every arm is BARE production plus its TLA -- no survey bag: the OFF baselines
# (p82vprod, p82bhoff) are bare production, and doc 82's wave 1 was voided by
# comparing a survey arm against a bare baseline.
#
# Usage:  WAVE=1 ./d83_arms.sh   (OFF gate both detectors + the distance sweep)
#         WAVE=2 ./d83_arms.sh   (p83vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p83/libpin_p83}
JOBS=${JOBS:-6}
LOGD=${LOGD:-/home/xqian/tmp/p83}

run () {  # run <arm> <det> <src> <extra-keys-or-empty>
    local arm=$1 det=$2 src=$3 extra=$4 tla=""
    [ -n "$extra" ] && tla="-S stm_michel_extra={$extra}" || tla=""
    echo "[$(date +%H:%M:%S)] $arm ($det) $extra"
    ARM=$arm DET=$det SRC=$src JOBS=$JOBS PIN=$PIN PR_TLA="$tla" \
        LOGD=$LOGD/arm_$arm "$X/d53_run_arms.sh" > "$LOGD/arm_${arm}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $arm rc=$?"
}

WAVE=${WAVE:-1}
if [ "$WAVE" = 1 ]; then
    run p83voff pdvd d16vnu "" &
    run p83hoff pdhd d16hnu "" &
    run p83v5   pdvd d16vnu "michel_near_stop_arm_cm:5.0" &
    run p83v7   pdvd d16vnu "michel_near_stop_arm_cm:7.0" &
    run p83v10  pdvd d16vnu "michel_near_stop_arm_cm:10.0" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p83vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

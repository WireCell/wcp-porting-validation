#!/usr/bin/env bash
# doc pdvd/84 -- the arms for doc 78 action item 3 (the moved-stop veto's reach exemption).
#
# Fork by duplication (CLAUDE.md M10) of d83_arms.sh.  The ONLY differences are
# the arm names, the TLAs and the pin.
#
# The pin is a FULL local/lib snapshot (the 572 top-level *.so*, dereferenced),
# not a Clus-only one: doc pdvd/80's wave 1 died with rc=139 at the ROOT-writing
# stage because a Clus-only pin was mixed with a rebuilt libWireCellRoot.
#
# Every arm is BARE production plus its TLA -- no survey bag: the OFF baselines
# (p83vprod, p83hoff) are bare production, and doc 82's wave 1 was voided by
# comparing a survey arm against a bare baseline.
#
# Usage:  WAVE=1 ./d84_arms.sh   (OFF gate both detectors + the reach sweep + the kink value)
#         WAVE=2 ./d84_arms.sh   (p84vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p84/libpin_p84}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p84}

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
    run p84voff pdvd d16vnu "" &
    run p84hoff pdhd d16hnu "" &
    run p84vr65 pdvd d16vnu "moved_stop_michel_reach_min_cm:6.5" &
    run p84vr8  pdvd d16vnu "moved_stop_michel_reach_min_cm:8.0" &
    run p84vr5  pdvd d16vnu "moved_stop_michel_reach_min_cm:5.0" &
    run p84vk59 pdvd d16vnu "moved_stop_michel_kink_min:59.1" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p84vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

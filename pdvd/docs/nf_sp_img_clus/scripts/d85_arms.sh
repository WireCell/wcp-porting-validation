#!/usr/bin/env bash
# doc pdvd/85 -- the arms for doc 78 action item 5 (the gamma collect's
# rejections; a capture gamma only on a stopper).
#
# Fork by duplication (CLAUDE.md M10) of d84_arms.sh.  The ONLY differences are
# the arm names, the TLAs and the pin.
#
# The pin is a FULL local/lib snapshot (the 572 top-level *.so*, dereferenced),
# not a Clus-only one: doc pdvd/80's wave 1 died with rc=139 at the ROOT-writing
# stage because a Clus-only pin was mixed with a rebuilt libWireCellRoot.
#
# Every gate arm is BARE production plus its TLA.  p85vwhs alone carries the
# d53 survey bag: it is the smoke run for the withheld segments' role-6 rej-16
# rows, graded by itself and compared against nothing.
#
# Usage:  WAVE=1 ./d85_arms.sh   (OFF gate both detectors + the knob + the 50 cm ring + the survey smoke)
#         WAVE=2 ./d85_arms.sh   (p85vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p85/libpin_p85}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p85}
SURVEY="survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0,publish_other_arms:true,segment_census:true"

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
    run p85voff  pdvd d16vnu "" &
    run p85hoff  pdhd d16hnu "" &
    run p85vwh   pdvd d16vnu "stop_gamma_require_stm:true" &
    run p85vsg50 pdvd d16vnu "stop_gamma_radius_cm:50.0" &
    run p85vwhs  pdvd d16vnu "stop_gamma_require_stm:true,$SURVEY" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p85vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

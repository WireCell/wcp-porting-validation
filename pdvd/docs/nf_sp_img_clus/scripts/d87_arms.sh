#!/usr/bin/env bash
# doc pdvd/87 -- the arms for doc 78 action item 7 (a size floor on doc 62 T3a's
# stop-local residual keep).
#
# Fork by duplication (CLAUDE.md M10) of d85_arms.sh.  The ONLY differences are
# the arm names, the TLAs, the pin and the wave split.
#
# The pin is a FULL local/lib snapshot (the 572 top-level *.so*, dereferenced),
# not a Clus-only one: doc pdvd/80's wave 1 died with rc=139 at the ROOT-writing
# stage because a Clus-only pin was mixed with a rebuilt libWireCellRoot.
#
# Every arm is BARE production plus its TLA (no d53 survey bag).
#
# Usage:  WAVE=1 ./d87_arms.sh   (OFF gate both detectors + the flip candidate)
#         WAVE=2 ./d87_arms.sh   (informational: radius 5 alone, radius 20 alone, radius 20 + floor)
#         WAVE=3 ./d87_arms.sh   (p87vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p87/libpin_p87}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p87}
FLOOR="stop_local_residual_min_points:5,stop_local_residual_min_len_cm:5.0"

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
    run p87voff  pdvd d16vnu "" &
    run p87hoff  pdhd d16hnu "" &
    run p87v5f   pdvd d16vnu "stop_local_residual_cm:5.0,$FLOOR" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p87v5    pdvd d16vnu "stop_local_residual_cm:5.0" &
    run p87v20   pdvd d16vnu "stop_local_residual_cm:20.0" &
    run p87v20f  pdvd d16vnu "stop_local_residual_cm:20.0,$FLOOR" &
    wait
fi
if [ "$WAVE" = 3 ]; then
    run p87vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

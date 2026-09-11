#!/usr/bin/env bash
# doc pdvd/88 -- the arms for doc 78 action item 7b (the stop snap restricted to
# what the entry reaches, on candidates where doc 87's stop-local keep fired).
#
# Fork by duplication (CLAUDE.md M10) of d87_arms.sh.  The ONLY differences are
# the arm names, the TLAs and the pin.
#
# The pin is a FULL local/lib snapshot (the 572 top-level *.so*, dereferenced),
# not a Clus-only one: doc pdvd/80's wave 1 died with rc=139 at the ROOT-writing
# stage because a Clus-only pin was mixed with a rebuilt libWireCellRoot.
#
# Every arm is BARE production plus its TLA (no d53 survey bag).
#
# Usage:  WAVE=1 ./d88_arms.sh   (OFF gate both detectors, the flip candidate, the knob alone)
#         WAVE=2 ./d88_arms.sh   (informational: radius 5 without the floor, radius 20 with it; both + the knob)
#         WAVE=3 ./d88_arms.sh   (p88vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p88/libpin_p88}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p88}
FLOOR="stop_local_residual_min_points:5,stop_local_residual_min_len_cm:5.0"
SNAP="stop_snap_reachable:true"

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
    run p88voff  pdvd d16vnu "" &
    run p88hoff  pdhd d16hnu "" &
    run p88v5fr  pdvd d16vnu "stop_local_residual_cm:5.0,$FLOOR,$SNAP" &
    run p88vr    pdvd d16vnu "$SNAP" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p88v5r   pdvd d16vnu "stop_local_residual_cm:5.0,$SNAP" &
    run p88v20fr pdvd d16vnu "stop_local_residual_cm:20.0,$FLOOR,$SNAP" &
    wait
fi
if [ "$WAVE" = 3 ]; then
    run p88vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

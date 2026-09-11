#!/usr/bin/env bash
# doc pdvd/90 -- the arms for doc 89's 0-FP bundle: P1's energy floor at 3 MeV
# (topology_michel_ke_min, C++ default 10, absent from the compiled config) and the
# plateau window's upper edge at 2.0 (plateau_mip_hi, compiled 1.6), together and alone.
#
# Fork by duplication (CLAUDE.md M10) of d88_arms.sh.  No C++ changes this round, so the
# binary is doc 88's full pin (libpin_p88, the one p88vprod ran on) and the baseline is
# p88vprod itself: same pin, same file, only the TLA differs.  No OFF gate is needed.
#
# Every arm is BARE production plus its TLA (no d53 survey bag).
#
# Usage:  WAVE=1 ./d90_arms.sh   (the bundle, and each half alone)
#         WAVE=2 ./d90_arms.sh   (p90vprod: the flipped file, no TLA)
set -uo pipefail
X=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
PIN=${PIN:-/home/xqian/tmp/p88/libpin_p88}
JOBS=${JOBS:-5}
LOGD=${LOGD:-/home/xqian/tmp/p90}

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
    run p90vb pdvd d16vnu "topology_michel_ke_min:3.0,plateau_mip_hi:2.0" &
    run p90vk pdvd d16vnu "topology_michel_ke_min:3.0" &
    run p90vp pdvd d16vnu "plateau_mip_hi:2.0" &
    wait
fi
if [ "$WAVE" = 2 ]; then
    run p90vprod pdvd d16vnu ""
fi
touch "$LOGD/DONE_wave$WAVE"
echo "[$(date +%H:%M:%S)] wave $WAVE done"

#!/usr/bin/env bash
# doc pdvd/75 (P1b) -- the arms, concurrently, each on its own pin, all BARE
# PRODUCTION config plus the knob under test (the doc 71 lesson: grade
# production claims on a bare-production arm, never on the survey TLA).
# Fork of d74_arms.sh.
#
# WAVE=1 (default), 6 arms x JOBS=4 = 24 of the 32 CPUs the owner licensed
# (a peer had 6 wire-cell jobs live at launch; loadavg checked after):
#   p75vleg / p75hleg : P74 pin (54e6ab99 = today's production binary) -> fresh baselines (p75vleg must equal p74vprod2)
#   p75voff / p75hoff : P75 pin, the new knob off                        -> the OFF gate
#   p75vfb            : bragg_anchor_geo_fallback (rise 1.5, the C++ default)
#   p75vfb0           : bragg_anchor_geo_fallback with bragg_anchor_rise_min 0 (the two-sided rule with no threshold: what the threshold buys)
# WAVE=2: ARMS2="label|extra-json-body;label|extra-json-body" (PDVD, P75 pin).
#
# Launch DETACHED (doc 74 sec 8: the harness killed a run_in_background arm):
#   nohup bash d75_arms.sh > /home/xqian/tmp/p75/arms_wave1.log 2>&1 < /dev/null & disown
# and wait on the ALL_DONE marker.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
B=/home/xqian/tmp/p74/libpin_p74
P=/home/xqian/tmp/p75/libpin_p75
W=/home/xqian/tmp/p75
J=${JOBS:-4}

run() {   # label det pin extra-json-body
    local lab=$1 det=$2 pin=$3 ext=$4 src=d16vnu
    [ "$det" = pdhd ] && src=d16hnu
    if [ -n "$ext" ]; then
        LOGD=$W/arm_$lab ARM=$lab DET=$det SRC=$src JOBS=$J PIN=$pin \
            PR_TLA="-S stm_michel_extra={$ext}" $R > $W/$lab.out 2>&1
    else
        LOGD=$W/arm_$lab ARM=$lab DET=$det SRC=$src JOBS=$J PIN=$pin $R > $W/$lab.out 2>&1
    fi
    echo "DONE $lab rc=$?"
}

case ${WAVE:-1} in
1)
    run p75vleg pdvd $B "" &
    run p75hleg pdhd $B "" &
    run p75voff pdvd $P "" &
    run p75hoff pdhd $P "" &
    run p75vfb  pdvd $P "bragg_anchor_geo_fallback:true" &
    run p75vfb0 pdvd $P "bragg_anchor_geo_fallback:true,bragg_anchor_rise_min:0.0" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $P "${x#*|}" & done
    ;;
esac
wait
echo ALL_DONE

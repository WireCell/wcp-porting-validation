#!/usr/bin/env bash
# docs pdvd/72 (P3b) + pdvd/73 (P2) -- the arms, concurrently, each on its own
# pin, all BARE PRODUCTION config plus the knob under test (the doc 71 lesson:
# grade production claims on a bare-production arm, never on the survey TLA).
#
# WAVE=1 (default), 7 arms x JOBS=4 = 28 of the 32 CPUs the owner licensed:
#   p72vleg / p72hleg : P4 pin (d227d5b8 = today's production binary)  -> fresh baselines (p72vleg must equal p4v50)
#   p72voff / p72hoff : P72 pin, every new knob off                      -> the OFF gate; its stop-arm DEBUG
#                                                                           lines (kink5, far_full) size P2 (b) and (c)
#   p72vb60 / p72vb90 : P3b at 60 / 90 deg                               -> the decision plateau (59.6, 132.6] deg:
#                                                                           the two must agree item for item
#   p72v2ab           : P3b 60 + P2 (a) 0.15 MIP when kink >= 60 + (b) shower far_len <= 60 cm
# WAVE=2: ARMS2="label|extra-json-body;label|extra-json-body" (PDVD, P72 pin),
#   chosen from the p72voff diagnostic -- doc 73 records which and why.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
B=/home/xqian/tmp/p4/libpin_p4
P=/home/xqian/tmp/p72/libpin_p72
W=/home/xqian/tmp/p72
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
}

B3='moved_stop_michel_kink_min:60.0'
case ${WAVE:-1} in
1)
    run p72vleg pdvd $B "" &
    run p72hleg pdhd $B "" &
    run p72voff pdvd $P "" &
    run p72hoff pdhd $P "" &
    run p72vb60 pdvd $P "$B3" &
    run p72vb90 pdvd $P "moved_stop_michel_kink_min:90.0" &
    run p72v2ab pdvd $P "$B3,michel_mip_lo_turned:0.15,michel_far_len_shower_max_cm:60.0" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $P "${x#*|}" & done
    ;;
esac
wait
echo ALL_DONE

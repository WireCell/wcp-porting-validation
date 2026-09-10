#!/usr/bin/env bash
# doc pdvd/74 (P3) -- the arms, concurrently, each on its own pin, all BARE
# PRODUCTION config plus the knob under test (the doc 71 lesson: grade
# production claims on a bare-production arm, never on the survey TLA).
# Fork of d72_arms.sh.
#
# WAVE=1 (default), 7 arms x JOBS=4 = 28 of the 32 CPUs the owner licensed:
#   p74vleg / p74hleg : P72 pin (299d8bc4 = today's production binary) -> fresh baselines (p74vleg must equal p72vprod)
#   p74voff / p74hoff : P74 pin, every new knob off                     -> the OFF gate
#   p74vcs            : michel_collinear_split (doc 70's P3 as proposed: retreat/split on a Bragg-confirmed chain)
#   p74vts            : retreat_tail_strict (the retreat's tail read past the vertex)
#   p74vtl            : retreat_tail_strict + retreat_tail_sublive (the tail also counts sub-live rows)
# WAVE=2: ARMS2="label|extra-json-body;label|extra-json-body" (PDVD, P74 pin).
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
B=/home/xqian/tmp/p72/libpin_p72
P=/home/xqian/tmp/p74/libpin_p74
W=/home/xqian/tmp/p74
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

case ${WAVE:-1} in
1)
    run p74vleg pdvd $B "" &
    run p74hleg pdhd $B "" &
    run p74voff pdvd $P "" &
    run p74hoff pdhd $P "" &
    run p74vcs  pdvd $P "michel_collinear_split:true" &
    run p74vts  pdvd $P "retreat_tail_strict:true" &
    run p74vtl  pdvd $P "retreat_tail_strict:true,retreat_tail_sublive:true" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $P "${x#*|}" & done
    ;;
esac
wait
echo ALL_DONE

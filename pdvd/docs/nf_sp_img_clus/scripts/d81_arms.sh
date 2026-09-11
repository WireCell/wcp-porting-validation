#!/usr/bin/env bash
# doc pdvd/81 -- the charge-based Michel energy (michel_q2d) and the Michel / STM
# 2-D cell table (michel_q2d_cells), the arms.  Bare PRODUCTION config (post
# doc 79 flip) plus the knobs under test.  Fork of d80_arms.sh.
#
# WAVE=1 (default), 6 arms x JOBS=4 = 24 of the 32 CPUs the owner licensed:
#   p81vleg / p81hleg : P80 pin (libpin_p80 = today's production binary, doc 80's census build) -> fresh baselines
#   p81voff / p81hoff : P81 pin, knobs off                                                       -> the OFF gate
#   p81vq2d / p81hq2d : michel_q2d:true,michel_q2d_cells:true                                    -> the ON arms
# WAVE=1b: the first wave's off/on arms ran on libpin_p81 (2525f189), whose estimator selected no
#   cells (the drift-frame defect of sec 5.1) -- wave 1b re-runs them on the FIXED pin libpin_p81f
#   under new names (nothing under work/ is overwritten): p81voff2 / p81hoff2 / p81vq2d2 / p81hq2d2.
#   The leg arms stand.
# WAVE=1c: wave 1b's estimator summed cross-shared cells raw (sec 5.2) and gave a charge-only Michel no
#   cells; the FINAL pin libpin_p81g fixes both -- wave 1c re-runs off/on under new names again:
#   p81voff3 / p81hoff3 / p81vq2d3 / p81hq2d3.  These are the graded arms.  Plus fresh leg arms
#   p81vleg2 / p81hleg2 on libpin_p75 (a FULL lib dir; wave 1's leg arms died in the tracking
#   visitor because libpin_p80 holds only libWireCellClus.so and the rest came from a local/lib
#   already rebuilt against the new TrackFitting layout -- a partial pin is not a pin).
# WAVE=2: ARMS2="label|extra-json-body;..." (PDVD, P81g pin).
# WAVE=3: the confirmation arms on the FLIPPED production files, no TLA: p81vprod / p81hprod.
#
# Launch DETACHED and wait on the ALL_DONE marker:
#   nohup bash d81_arms.sh > /home/xqian/tmp/p81/arms_wave1.log 2>&1 < /dev/null & disown
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
B=/home/xqian/tmp/p80/libpin_p80       # wave 1 leg (a PARTIAL pin: clus only -- the leg arms died; superseded)
L=/home/xqian/tmp/p75/libpin_p75       # wave 1c leg: the last full pre-change pin (toolkit 567a7232)
P=/home/xqian/tmp/p81/libpin_p81      # wave 1 (superseded)
F=/home/xqian/tmp/p81/libpin_p81f     # wave 1b (superseded)
G=/home/xqian/tmp/p81/libpin_p81g     # the final pin, waves 1c/2/3
W=/home/xqian/tmp/p81
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
    run p81vleg pdvd $B "" &
    run p81hleg pdhd $B "" &
    run p81voff pdvd $P "" &
    run p81hoff pdhd $P "" &
    run p81vq2d pdvd $P "michel_q2d:true,michel_q2d_cells:true" &
    run p81hq2d pdhd $P "michel_q2d:true,michel_q2d_cells:true" &
    ;;
1b)
    run p81voff2 pdvd $F "" &
    run p81hoff2 pdhd $F "" &
    run p81vq2d2 pdvd $F "michel_q2d:true,michel_q2d_cells:true" &
    run p81hq2d2 pdhd $F "michel_q2d:true,michel_q2d_cells:true" &
    ;;
1c)
    run p81vleg2 pdvd $L "" &
    run p81hleg2 pdhd $L "" &
    run p81voff3 pdvd $G "" &
    run p81hoff3 pdhd $G "" &
    run p81vq2d3 pdvd $G "michel_q2d:true,michel_q2d_cells:true" &
    run p81hq2d3 pdhd $G "michel_q2d:true,michel_q2d_cells:true" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $G "${x#*|}" & done
    ;;
3)
    run p81vprod pdvd $G "" &
    run p81hprod pdhd $G "" &
    ;;
esac
wait
echo ALL_DONE

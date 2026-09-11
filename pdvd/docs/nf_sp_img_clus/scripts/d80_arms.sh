#!/usr/bin/env bash
# doc pdvd/80 -- the segment census (segment_census, role 8), the arms.  Bare
# PRODUCTION config (post doc 79 flip) on the P80 pin (libpin_p80 = the census
# build), plus the one knob under test.  Fork of d79_arms.sh.
#
# WAVE=1 (default), 3 arms x JOBS=4:
#   p80voff / p80hoff : P80 pin, knob off  -> the OFF gate (PDVD vs p79vprod, PDHD vs p75hoff)
#   p80vcen           : segment_census:true (PDVD)
# WAVE=2: ARMS2="label|extra-json-body;..." (PDVD, P80 pin).
# WAVE=3: the same three arms on the FULL pin libpin_p80full (every local/lib .so,
#   snapshot 19:05 2026-09-10, Clus 2525f189 / Root 46bf5105), labels p80boff /
#   p80bhoff / p80bcen.  Why: a concurrent session reinstalled seven libraries under
#   wave 1 (18:58), and a Clus-only pin against the new libWireCellRoot then crashed
#   every job at the ROOT-writing stage (rc=139) -- the wave-1 PDVD arms are void
#   (108/109 of 120 complete, the last 12 events of run 039349 dead) and are kept
#   only as the record of the failure.
#
# Launch DETACHED and wait on the ALL_DONE marker:
#   nohup bash d80_arms.sh > /home/xqian/tmp/p80/arms_wave1.log 2>&1 < /dev/null & disown
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
P=/home/xqian/tmp/p80/libpin_p80
W=/home/xqian/tmp/p80
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
    run p80voff pdvd $P "" &
    run p80hoff pdhd $P "" &
    run p80vcen pdvd $P "segment_census:true" &
    ;;
3)
    P=/home/xqian/tmp/p80/libpin_p80full
    run p80boff pdvd $P "" &
    run p80bhoff pdhd $P "" &
    run p80bcen pdvd $P "segment_census:true" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $P "${x#*|}" & done
    ;;
esac
wait
echo ALL_DONE

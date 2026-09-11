#!/usr/bin/env bash
# doc pdvd/79 -- CheckSTM_Michel's max_candidates cap (doc 77 sec 7.1), the arms.
# Bare PRODUCTION config on the production pin (P75, libpin_p75 = toolkit
# 567a7232, md5 02557b8d), plus the one value under test.  Fork of d75_arms.sh.
#
# WAVE=1 (default):
#   p79vcap : max_candidates 64 (C++ default 8; the cap fires on 6 of 120 PDVD
#             events with 9-13 candidates, so 64 is "no cap" on this sample)
# WAVE=2: ARMS2="label|extra-json-body;label|extra-json-body" (PDVD, P75 pin),
#   e.g. the confirmation arm after the flip: ARMS2="p79vprod|" (no extra).
#
# Launch DETACHED and wait on the ALL_DONE marker:
#   nohup bash d79_arms.sh > /home/xqian/tmp/p79/arms_wave1.log 2>&1 < /dev/null & disown
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
P=/home/xqian/tmp/p75/libpin_p75
W=/home/xqian/tmp/p79
J=${JOBS:-8}

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
    run p79vcap pdvd $P "max_candidates:64" &
    ;;
2)
    IFS=';' read -ra ARMLIST <<< "${ARMS2:?set ARMS2 for wave 2}"
    for x in "${ARMLIST[@]}"; do run "${x%%|*}" pdvd $P "${x#*|}" & done
    ;;
esac
wait
echo ALL_DONE

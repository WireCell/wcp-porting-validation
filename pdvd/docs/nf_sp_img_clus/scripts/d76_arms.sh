#!/usr/bin/env bash
# doc pdvd/76 (P5) -- the two smoke arms for stm_trackfitting_config_file, BARE
# PRODUCTION config, the production pin (P75 after doc 75's flip), 2 arms x JOBS=8.
#   p76vsame : -A stm_trackfitting_config=/home/xqian/tmp/p76/tf_same.json
#              a byte-identical copy of pdvd_track_fitting.json at an absolute path:
#              proves the TLA reaches TaggerCheckSTM + CheckSTM_Michel and changes
#              nothing else (every output identical to the bare baseline)
#   p76vdx4  : -A stm_trackfitting_config=/home/xqian/tmp/p76/tf_dx04.json
#              the same file with dx_norm_length 4 mm: the effect is visible on the
#              STM side and the neutrino path is untouched (doc 68 sec 4 could not
#              scope this).  Reported, never flipped (doc 70: a step change needs a
#              new scan).
# Launch DETACHED (doc 74 sec 8):
#   nohup bash d76_arms.sh > /home/xqian/tmp/p76/arms.log 2>&1 < /dev/null & disown
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
P=${PIN:-/home/xqian/tmp/p75/libpin_p75}
W=/home/xqian/tmp/p76
J=${JOBS:-8}

run() {   # label tla
    local lab=$1 tla=$2
    LOGD=$W/arm_$lab ARM=$lab DET=pdvd SRC=d16vnu JOBS=$J PIN=$P PR_TLA="$tla" $R > $W/$lab.out 2>&1
    echo "DONE $lab rc=$?"
}
run p76vsame "-A stm_trackfitting_config=$W/tf_same.json" &
run p76vdx4  "-A stm_trackfitting_config=$W/tf_dx04.json" &
wait
echo ALL_DONE

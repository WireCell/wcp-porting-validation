#!/usr/bin/env bash
# doc pdvd/70 sec 10 -- the six P1 arms, concurrently, each on its own pin.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
S='survey_enable:true,survey_radius_cm:60.0,survey_max_len_cm:25.0'
B=/home/xqian/tmp/d71/libpin_base; P=/home/xqian/tmp/d71/libpin_p1
export LOGD
LOGD=/home/xqian/tmp/d71/arm_d71vleg ARM=d71vleg DET=pdvd SRC=d16vnu JOBS=5 PIN=$B PR_TLA="-S stm_michel_extra={$S}" $R > /home/xqian/tmp/d71/d71vleg.out 2>&1 &
LOGD=/home/xqian/tmp/d71/arm_d71hleg ARM=d71hleg DET=pdhd SRC=d16hnu JOBS=5 PIN=$B PR_TLA="-S stm_michel_extra={$S}" $R > /home/xqian/tmp/d71/d71hleg.out 2>&1 &
LOGD=/home/xqian/tmp/d71/arm_d71voff ARM=d71voff DET=pdvd SRC=d16vnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={$S}" $R > /home/xqian/tmp/d71/d71voff.out 2>&1 &
LOGD=/home/xqian/tmp/d71/arm_d71hoff ARM=d71hoff DET=pdhd SRC=d16hnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={$S}" $R > /home/xqian/tmp/d71/d71hoff.out 2>&1 &
LOGD=/home/xqian/tmp/d71/arm_d71vp1 ARM=d71vp1 DET=pdvd SRC=d16vnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={$S,topology_stop_evidence:true}" $R > /home/xqian/tmp/d71/d71vp1.out 2>&1 &
LOGD=/home/xqian/tmp/d71/arm_d71vsp ARM=d71vsp DET=pdvd SRC=d16vnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={$S,topology_stop_evidence:true,topology_clears_sparse:true}" $R > /home/xqian/tmp/d71/d71vsp.out 2>&1 &
wait
echo ALL_DONE

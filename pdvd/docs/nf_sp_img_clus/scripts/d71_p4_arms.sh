#!/usr/bin/env bash
# doc pdvd/71 -- the six P4 arms, concurrently, each on its own pin (fork of
# d70_p1_arms.sh).  Unlike the P1 arms these run BARE PRODUCTION config (no
# survey TLA): the flip reproduces production, and the survey's 60 cm admission
# is itself a fit perturbation (doc pdvd/53 sec 6.2).
#   p4vleg / p4hleg : P1 pin (f66a8b9f = today's production binary), no TLA
#   p4voff / p4hoff : P4 pin, knob off              -> the OFF gate
#   p4v35           : P4 pin, michel_gamma_collect   -> the flip candidate (35 cm, admission unchanged)
#   p4v60           : P4 pin, + radius 60 cm         -> widened admission, evidence only
# 6 arms x JOBS=5 = 30 jobs on the 32 CPUs the owner licensed.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
R=pdvd/docs/nf_sp_img_clus/scripts/d53_run_arms.sh
B=/home/xqian/tmp/d71/libpin_p1; P=/home/xqian/tmp/p4/libpin_p4
W=/home/xqian/tmp/p4
export LOGD
LOGD=$W/arm_p4vleg ARM=p4vleg DET=pdvd SRC=d16vnu JOBS=5 PIN=$B $R > $W/p4vleg.out 2>&1 &
LOGD=$W/arm_p4hleg ARM=p4hleg DET=pdhd SRC=d16hnu JOBS=5 PIN=$B $R > $W/p4hleg.out 2>&1 &
LOGD=$W/arm_p4voff ARM=p4voff DET=pdvd SRC=d16vnu JOBS=5 PIN=$P $R > $W/p4voff.out 2>&1 &
LOGD=$W/arm_p4hoff ARM=p4hoff DET=pdhd SRC=d16hnu JOBS=5 PIN=$P $R > $W/p4hoff.out 2>&1 &
LOGD=$W/arm_p4v35 ARM=p4v35 DET=pdvd SRC=d16vnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={michel_gamma_collect:true}" $R > $W/p4v35.out 2>&1 &
LOGD=$W/arm_p4v60 ARM=p4v60 DET=pdvd SRC=d16vnu JOBS=5 PIN=$P PR_TLA="-S stm_michel_extra={michel_gamma_collect:true,michel_gamma_radius_cm:60.0}" $R > $W/p4v60.out 2>&1 &
wait
echo ALL_DONE

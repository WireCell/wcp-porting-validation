#!/bin/bash
# prod0923: nuecc48 + ncpi0 (reality=data) at ref/prod-2026-09-21d, toolkit 377119ee, pinned binary
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin || exit 9
export LD_LIBRARY_PATH=$HOME/tmp/prod0923-libpin:$LD_LIBRARY_PATH
R1=input_files_reco1
run_sample() {
  s=$1; shift
  SBND_MAX_JOBS=$MJ ./run_chain_group.sh "$@" > .prod0923-$s-stageA.log 2>&1
  echo "stageA $s rc=$?" >> .prod0923-status
  PR_EXTRA_STAGES=pr_display PR_JOBS=$PJ ./run_pr_chain_batch.sh work-$s-prod0923 work-$s-prod0923pr data > .prod0923-$s-stageB.log 2>&1
  echo "stageB $s rc=$?" >> .prod0923-status
}
MJ=3 PJ=8 run_sample nuecc48 $R1/data_filtered_decoded_reco1-fe6033f3-07a0-4971-cea5-16ce59269fba_eventidfiltered_frameshift.root work-nuecc48-prod0923 data --size 16 --layout perevt &
MJ=2 PJ=6 run_sample ncpi0 $R1/nc-sideband_filtered_frameshift.root work-ncpi0-prod0923 data --size 16 --layout perevt --fsproduct 'sbnd::timing::FrameShiftInfo_frameshift__FILTERFRAMESHIFT.' &
wait
echo "ALL DONE" >> .prod0923-status

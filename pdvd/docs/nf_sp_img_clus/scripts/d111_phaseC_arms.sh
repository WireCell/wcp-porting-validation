#!/usr/bin/env bash
# doc pdvd/111 Phase C driver: base (knob absent) + 3 seed_recenter_sigma arms per detector, concurrently (shared load,
# so the pre-registered wall clause R compares arms under the same machine load).  Run with nohup; logs in /home/xqian/tmp/d111.
set -u
S=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/scripts
F=/home/xqian/toolkit-dev/wcp-porting-img/pdvd/docs/nf_sp_img_clus/figs
PIN=/home/xqian/tmp/d111/libpin_d111b
echo "phaseC start $(date -Is) pred $(cut -c1-16 $F/111_pred.sha256)"
ARM=d111hoff DET=pdhd SRC=d108hflip JOBS=2 PIN=$PIN bash $S/d111_run_arms.sh > /home/xqian/tmp/d111/arm_d111hoff.out 2>&1 &
for s in 6 10 15; do
  ARM=d111hsr$s DET=pdhd SRC=d108hflip JOBS=2 PIN=$PIN PR_TLA="-A trackfitting_config=$F/111_tf_pdhd_sr$s.json" bash $S/d111_run_arms.sh > /home/xqian/tmp/d111/arm_d111hsr$s.out 2>&1 &
done
ARM=d111voff DET=pdvd SRC=d103vflip JOBS=2 PIN=$PIN bash $S/d111_run_arms.sh > /home/xqian/tmp/d111/arm_d111voff.out 2>&1 &
for s in 6 10 15; do
  ARM=d111vsr$s DET=pdvd SRC=d103vflip JOBS=2 PIN=$PIN PR_TLA="-A trackfitting_config=$F/111_tf_pdvd_sr$s.json" bash $S/d111_run_arms.sh > /home/xqian/tmp/d111/arm_d111vsr$s.out 2>&1 &
done
wait
echo "phaseC done $(date -Is)"
echo DRIVE_C_DONE

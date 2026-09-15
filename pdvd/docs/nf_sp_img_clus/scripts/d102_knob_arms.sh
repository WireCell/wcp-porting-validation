#!/usr/bin/env bash
# doc pdvd/102 P4 -- the simulation arms, one detector per call.  Fork by duplication of d101_knob_arms.sh
# (untouched).  Pre-registered rule: /home/xqian/tmp/d102/pred.txt (sha256 in pred.sha256), frozen BEFORE these ran.
#   ISO set (d102_sim_pr_arm.sh, runs 900103/900104):   d102b, d102brep, d102cs1, d102cs2, d102cs3
#   low set (d101_sim_pr_arm.sh, runs 900101/900102):   d102lcs1, d102lcs2, d102lcs3   (baseline = doc 101 d101kf)
# Usage: DET=pdvd|pdhd PIN=/home/xqian/tmp/d102/libpin_d102 [JOBS=2] [ARMS="..."] bash d102_knob_arms.sh
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
S=$IMG/pdvd/docs/nf_sp_img_clus/scripts
F=$IMG/pdvd/docs/nf_sp_img_clus/figs
DET=${DET:?set DET}
PIN=${PIN:?set PIN}
JOBS=${JOBS:-2}
ARMS=${ARMS:-"d102b d102brep d102cs1 d102cs2 d102cs3 d102lcs1 d102lcs2 d102lcs3"}
TF=$F/101_tf_sim_${DET}_kf.json
CS="-S retile_sampler_strategy='charge_stepped'"
md5_before=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[d102 arms] det=$DET pin=$PIN clus md5 $md5_before arms=<$ARMS> pred $(cut -c1-16 /home/xqian/tmp/d102/pred.sha256 | head -1) $(date -Is)"
for arm in $ARMS; do
    case "$arm" in
        d102b|d102brep) script=d102_sim_pr_arm.sh; extra="" ;;
        d102cs1) script=d102_sim_pr_arm.sh; extra="$CS" ;;
        d102cs2) script=d102_sim_pr_arm.sh; extra="$CS -S retile_sampler_charge_threshold=2000" ;;
        d102cs3) script=d102_sim_pr_arm.sh; extra="$CS -S retile_sampler_wire_product=10000" ;;
        # round 2 (pred2.txt): the fit-knob-OFF cells, sim-matched JSON without the knob keys
        d102o)    script=d102_sim_pr_arm.sh; extra="-S retile_sampler_strategy='stepped'"; TF=$F/101_tf_sim_${DET}.json ;;
        d102ocs)  script=d102_sim_pr_arm.sh; extra="$CS"; TF=$F/101_tf_sim_${DET}.json ;;
        d102locs) script=d101_sim_pr_arm.sh; extra="$CS"; TF=$F/101_tf_sim_${DET}.json ;;
        d102lcs1) script=d101_sim_pr_arm.sh; extra="$CS" ;;
        d102lcs2) script=d101_sim_pr_arm.sh; extra="$CS -S retile_sampler_charge_threshold=2000" ;;
        d102lcs3) script=d101_sim_pr_arm.sh; extra="$CS -S retile_sampler_wire_product=10000" ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    echo "[$(date +%H:%M:%S)] $arm script=$script tf=$(basename "$TF") extra=<${extra:-none}>"
    ARM=$arm DET=$DET PIN=$PIN JOBS=$JOBS TF=$TF EXTRA_TLA="$extra" bash "$S/$script"
    echo "[$(date +%H:%M:%S)] $arm rc=$?  loadavg $(cut -d' ' -f1 /proc/loadavg)"
done
md5_after=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[d102 arms] pin md5 after $md5_after"
[ "$md5_before" = "$md5_after" ] || echo "*** PIN CHANGED MID-CAMPAIGN -- every arm is unattributable ***"
echo KNOB_ARMS_DONE

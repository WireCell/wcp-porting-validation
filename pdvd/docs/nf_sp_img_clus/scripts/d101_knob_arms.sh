#!/usr/bin/env bash
# doc pdvd/101 sec 4.0 / 4.4 -- the knob arms on the simulated muons, one detector per call.
#
# Pre-registered (sec 4.0, /home/xqian/tmp/d101/pred.txt, sha256 0c67b8ae...) BEFORE these ran:
#   d101koff  new pin, every knob at its default      -> must equal d101b bit for bit (the
#             knob-off runtime gate on the simulation chain; d101b ran the pre-knob pin libpin_new)
#   d101kf    fit knobs:     fit_weight_pow 1.5, assoc_cont_center 1   (figs/101_tf_sim_<det>_kf.json)
#   d101ks    retile sampler: retile_sampler_half_pitch true, retile_sampler_min_step 1
#   d101kc    both
# Every arm symlinks the same _d101 pctree (d101_sim_pr_arm.sh) and refuses an existing tag (M13).
#
# Usage: DET=pdvd|pdhd PIN=/home/xqian/tmp/d101/libpin_k [JOBS=4] [ARMS="d101koff d101kf d101ks d101kc"] \
#        bash d101_knob_arms.sh > /home/xqian/tmp/d101/knob_arms_<det>.log 2>&1
set -uo pipefail
IMG=/home/xqian/toolkit-dev/wcp-porting-img
S=$IMG/pdvd/docs/nf_sp_img_clus/scripts
F=$IMG/pdvd/docs/nf_sp_img_clus/figs
DET=${DET:?set DET}
PIN=${PIN:?set PIN}
JOBS=${JOBS:-4}
ARMS=${ARMS:-"d101koff d101kf d101ks d101kc"}
SAMP="-S retile_sampler_half_pitch=true -S retile_sampler_min_step=1"
md5_before=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[d101 knob arms] det=$DET pin=$PIN clus md5 $md5_before arms=<$ARMS> $(date -Is)"
for arm in $ARMS; do
    case "$arm" in
        d101koff) tf=$F/101_tf_sim_${DET}.json;    extra="" ;;
        d101kf)   tf=$F/101_tf_sim_${DET}_kf.json; extra="" ;;
        d101ks)   tf=$F/101_tf_sim_${DET}.json;    extra="$SAMP" ;;
        d101kc)   tf=$F/101_tf_sim_${DET}_kf.json; extra="$SAMP" ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    echo "[$(date +%H:%M:%S)] $arm tf=$(basename "$tf") extra=<${extra:-none}>"
    ARM=$arm DET=$DET PIN=$PIN JOBS=$JOBS TF=$tf EXTRA_TLA="$extra" bash "$S/d101_sim_pr_arm.sh"
    echo "[$(date +%H:%M:%S)] $arm rc=$?  loadavg $(cut -d' ' -f1 /proc/loadavg)"
done
md5_after=$(md5sum "$PIN/libWireCellClus.so" | cut -c1-12)
echo "[d101 knob arms] pin md5 after $md5_after"
[ "$md5_before" = "$md5_after" ] || echo "*** PIN CHANGED MID-CAMPAIGN -- every arm is unattributable ***"
echo KNOB_ARMS_DONE

#!/bin/bash
# doc pdvd/102 P2 -- counterfactual PR arms on the owner's PDHD event and the two PDVD examples.
# dbg = production -nu -stm-fit + knob-on fit JSON (== d101smkf / d101vkf config) + log-only debug envs
# cs  = the same + retile charge_stepped (prototype defaults).  Same pctree (symlinks), same pin.
set -u
IMG=/home/xqian/toolkit-dev/wcp-porting-img
PIN=/home/xqian/tmp/d102/libpin_d102
F=$IMG/pdvd/docs/nf_sp_img_clus/figs
L=/home/xqian/tmp/d102/logs; mkdir -p $L
echo "[cf] start $(date -Is) pin clus md5 $(md5sum $PIN/libWireCellClus.so | cut -c1-12) loadavg $(cut -d' ' -f1 /proc/loadavg)"
run_one() {  # det run evt arm tla
  local det=$1 run=$2 evt=$3 arm=$4 tla=$5 UP; UP=$(echo $det | tr a-z A-Z)
  local extra=""; [ $det = pdvd ] && extra="PDVD_LIGHT_SUFFIX=_keep"
  ( cd $IMG/$det && env LD_LIBRARY_PATH=$PIN WCT_PYLIB=off WCT_STM_PATH_DEBUG=1 WCT_STEINER_PHASE_DUMP=1 $extra \
        ${UP}_PR_TLA="$tla" ./run_pr_evt.sh -nu -stm-fit -s $arm $run $evt ) > $L/${det}_${run}_${evt}_${arm}.out 2>&1
  echo "[cf] $det $run/$evt $arm rc=$? $(date +%H:%M:%S)"
}
HTF="-A trackfitting_config=$F/101_tf_prod_pdhd_kf.json"
VTF="-A trackfitting_config=$F/101_tf_prod_pdvd_kf.json"
CS="-S retile_sampler_strategy='charge_stepped'"
run_one pdhd 28084 21 d102dbg "$HTF" &
run_one pdhd 28084 21 d102cs  "$HTF $CS" &
run_one pdvd 39253 8  d102vdbg "$VTF" &
run_one pdvd 39253 8  d102vcs  "$VTF $CS" &
run_one pdvd 39349 64 d102vdbg "$VTF" &
run_one pdvd 39349 64 d102vcs  "$VTF $CS" &
wait
echo "[cf] pin clus md5 after $(md5sum $PIN/libWireCellClus.so | cut -c1-12)"
echo CF_DONE
